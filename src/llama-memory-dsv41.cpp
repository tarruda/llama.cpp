#include "llama-memory-dsv41.h"

#include "llama-batch.h"
#include "llama-io.h"
#include "llama-model.h"
#include "llama-impl.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "../ggml/src/ggml-quants.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

llama_memory_dsv41::llama_memory_dsv41(const llama_model & model, uint32_t n_ctx, uint32_t n_seq_max, uint32_t n_ubatch, bool offload) :
    hparams(model.hparams), n_ctx(n_ctx), n_ring(std::min(n_ctx, hparams.n_swa + n_ubatch)), n_vocab(model.vocab.n_tokens()) {
    if (n_seq_max != 1 || n_ctx == 0 || n_ring < hparams.n_swa) {
        throw std::runtime_error("DeepSeek-V4.1 currently requires one sequence and a context at least as large as the sliding window");
    }
    std::map<ggml_backend_buffer_type_t, ggml_context_ptr> ctx_map;
    const auto context_for = [&](ggml_backend_buffer_type_t buft) {
        auto & ctx = ctx_map[buft];
        if (!ctx) {
            const ggml_init_params params = { ggml_tensor_overhead()*(hparams.n_layer()*5 + 2), nullptr, true };
            ctx.reset(ggml_init(params));
            if (!ctx) { throw std::runtime_error("failed to create DeepSeek-V4.1 state context"); }
        }
        return ctx.get();
    };
    auto * cpu_ctx = context_for(ggml_backend_cpu_buffer_type());
    layers.resize(hparams.n_layer());
    const auto create = [&](ggml_context * ctx, ggml_type type, int64_t width, int64_t rows, const char * name, int il) {
        auto * tensor = ggml_new_tensor_2d(ctx, type, width, rows);
        ggml_format_name(tensor, "dsv41_%s_l%d", name, il);
        return tensor;
    };
    size_t global_bytes = 0, ring_bytes = 0, compressor_bytes = 0;
    for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
        auto & layer = layers[il];
        const int64_t dim = hparams.n_embd_head_k(il);
        auto * cache_ctx = cpu_ctx;
        auto * dev = model.dev_layer(il);
        if (offload && std::strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev)), "MTL") == 0) {
            cache_ctx = context_for(ggml_backend_dev_buffer_type(dev));
        }
        if (il == hparams.n_layer()/2) {
            encoder_hidden = create(cache_ctx, GGML_TYPE_BF16, hparams.n_embd*hparams.dsv4_hc_mult, n_ring, "encoder_hidden", il);
            encoder_pre = create(cache_ctx, GGML_TYPE_F32, hparams.dsv4_hc_mult, n_ring, "encoder_pre", il);
        }
        layer.raw = create(cache_ctx, GGML_TYPE_I8, dim/32*33, n_ring, "raw", il);
        ring_bytes += ggml_nbytes(layer.raw);
        if (hparams.dsv41_kv_source[il] != (int32_t) il) {
            continue;
        }
        const auto ratio = hparams.dsv4_compress_ratios[il];
        const uint32_t rows = (n_ctx + ratio - 1)/ratio;
        layer.kv = create(cache_ctx, GGML_TYPE_I8, dim/64*sizeof(block_nvfp4), rows, "kv", il);
        layer.index = create(cache_ctx, GGML_TYPE_I8, hparams.indexer_head_size/32*sizeof(block_mxfp4), rows, "index", il);
        global_bytes += ggml_nbytes(layer.kv) + ggml_nbytes(layer.index);
        if (ratio > 1) {
            layer.comp_kv = create(cache_ctx, GGML_TYPE_F32, dim, n_ring, "comp_kv", il);
            layer.comp_score = create(cache_ctx, GGML_TYPE_F32, dim, n_ring, "comp_score", il);
            compressor_bytes += ggml_nbytes(layer.comp_kv) + ggml_nbytes(layer.comp_score);
        }
    }
    for (auto & [buft, ctx] : ctx_map) {
        if (!ggml_get_first_tensor(ctx.get())) { continue; }
        ggml_backend_buffer_ptr buffer;
        if (hparams.no_alloc) {
            buffer.reset(ggml_backend_buft_alloc_buffer(buft, 0));
            for (auto * tensor = ggml_get_first_tensor(ctx.get()); tensor; tensor = ggml_get_next_tensor(ctx.get(), tensor)) {
                tensor->buffer = buffer.get();
            }
        } else {
            buffer.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft));
        }
        if (!buffer) { throw std::runtime_error("failed to allocate DeepSeek-V4.1 state"); }
        LLAMA_LOG_INFO("%s: %s state buffer = %.2f MiB\n", __func__, ggml_backend_buft_name(buft), ggml_backend_buffer_get_size(buffer.get())/1048576.0);
        ctxs_bufs.emplace_back(std::move(ctx), std::move(buffer));
    }
    clear(true);
    LLAMA_LOG_INFO("%s: global KV/index = %.2f MiB, SWA/rollback = %.2f MiB, compressor/rollback = %.2f MiB\n", __func__, global_bytes/1048576.0, ring_bytes/1048576.0, compressor_bytes/1048576.0);
    LLAMA_LOG_INFO("%s: encoder frontier/rollback = %.2f MiB\n", __func__, (ggml_nbytes(encoder_hidden) + ggml_nbytes(encoder_pre))/1048576.0);
}

llama_memory_context_ptr llama_memory_dsv41::init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) {
    GGML_UNUSED(embd_all);
    if (!prefill && !decoder_ready) {
        LLAMA_LOG_ERROR("%s: decoder state is incomplete; finish CED prefill with a final output request\n", __func__);
        return std::make_unique<llama_memory_dsv41_context>(this, std::vector<llama_ubatch>{}, LLAMA_MEMORY_STATUS_FAILED_PREPARE);
    }
    balloc.split_reset();
    std::vector<llama_ubatch> ubatches;
    size_t position = tokens.size();
    while (true) {
        auto ubatch = balloc.split_simple(n_ubatch);
        if (ubatch.n_tokens == 0) {
            break;
        }
        bool valid = ubatch.token && position + ubatch.n_tokens <= n_ctx;
        for (uint32_t i = 0; valid && i < ubatch.n_tokens; ++i) {
            valid = ubatch.pos[i] == (llama_pos) (position + i) && ubatch.n_seq_id[i] == 1 && ubatch.seq_id[i][0] == 0;
        }
        if (!valid) {
            LLAMA_LOG_ERROR("%s: expected contiguous token IDs for sequence 0 within the configured context\n", __func__);
            return std::make_unique<llama_memory_dsv41_context>(this, std::vector<llama_ubatch>{}, LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }
        position += ubatch.n_tokens;
        ubatches.push_back(std::move(ubatch));
    }
    const auto status = balloc.get_n_used() == balloc.get_n_tokens() ? LLAMA_MEMORY_STATUS_SUCCESS : LLAMA_MEMORY_STATUS_FAILED_PREPARE;
    return std::make_unique<llama_memory_dsv41_context>(this, std::move(ubatches), status);
}

llama_memory_context_ptr llama_memory_dsv41::init_full() {
    return std::make_unique<llama_memory_dsv41_context>(this, std::vector<llama_ubatch>{}, LLAMA_MEMORY_STATUS_SUCCESS);
}

llama_memory_context_ptr llama_memory_dsv41::init_update(llama_context * lctx, bool optimize) {
    GGML_UNUSED(lctx);
    GGML_UNUSED(optimize);
    return std::make_unique<llama_memory_dsv41_context>(this, std::vector<llama_ubatch>{}, LLAMA_MEMORY_STATUS_NO_UPDATE);
}

bool llama_memory_dsv41::apply(const llama_ubatch & ubatch) {
    if (!ubatch.n_tokens || !ubatch.token || !ubatch.pos || ubatch.pos[0] != (llama_pos) tokens.size() || tokens.size() + ubatch.n_tokens > n_ctx) {
        return false;
    }
    tokens.insert(tokens.end(), ubatch.token, ubatch.token + ubatch.n_tokens);
    ring_end = std::max<uint32_t>(ring_end, tokens.size());
    decoder_ready = false;
    return true;
}

void llama_memory_dsv41::complete(const llama_ubatch & ubatch, bool success) {
    if (!success) {
        // CED replay can overwrite decoder rows before the failed ubatch.
        if (!prefill && tokens.size() == (size_t) ubatch.pos[0]) { decoder_ready = true; }
        return;
    }
    const bool output = std::any_of(ubatch.output, ubatch.output + ubatch.n_tokens, [](int8_t value) { return value != 0; });
    if (!prefill || output) {
        if (prefill) { decoder_start = tokens.size() - std::min<size_t>(tokens.size(), hparams.n_swa); }
        decoder_ready = true;
    }
}

void llama_memory_dsv41::clear(bool data) {
    tokens.clear();
    ring_end = 0;
    decoder_start = 0;
    decoder_ready = true;
    if (data && !hparams.no_alloc) {
        for (auto & entry : ctxs_bufs) { ggml_backend_buffer_clear(entry.second.get(), 0); }
    }
}

bool llama_memory_dsv41::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    if (seq_id != 0 && seq_id != -1) {
        return true;
    }
    p0 = std::max<llama_pos>(0, p0);
    p1 = p1 < 0 ? tokens.size() : std::min<size_t>(p1, tokens.size());
    if (p0 >= p1) {
        return true;
    }
    if (p1 != (llama_pos) tokens.size()) {
        return false;
    }
    if (p0 != 0 && ring_end > n_ring && p0 < (int64_t) ring_end - n_ring + hparams.n_swa) {
        return false;
    }
    if (p0 != 0 && decoder_ready && p0 < (llama_pos) decoder_start) { return false; }
    tokens.resize(p0);
    decoder_start = std::min<uint32_t>(decoder_start, p0);
    if (p0 == 0) {
        ring_end = 0;
        decoder_ready = true;
    }
    return true;
}

void llama_memory_dsv41::seq_cp(llama_seq_id src, llama_seq_id dst, llama_pos p0, llama_pos p1) {
    GGML_UNUSED(p0);
    GGML_UNUSED(p1);
    GGML_ASSERT(src == 0 && dst == 0);
}

void llama_memory_dsv41::seq_keep(llama_seq_id seq_id) {
    if (seq_id != 0) {
        clear(false);
    }
}

void llama_memory_dsv41::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    GGML_UNUSED(seq_id);
    GGML_UNUSED(p0);
    GGML_UNUSED(p1);
    GGML_ASSERT(shift == 0);
}

void llama_memory_dsv41::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    GGML_UNUSED(seq_id);
    GGML_UNUSED(p0);
    GGML_UNUSED(p1);
    GGML_ASSERT(d == 1);
}

llama_pos llama_memory_dsv41::seq_pos_min(llama_seq_id seq_id) const {
    if (seq_id != 0 || tokens.empty()) { return -1; }
    return std::max(ring_end > n_ring ? ring_end - n_ring : 0u, decoder_ready ? decoder_start : 0u);
}

llama_pos llama_memory_dsv41::seq_pos_max(llama_seq_id seq_id) const {
    return seq_id == 0 && !tokens.empty() ? (llama_pos) tokens.size() - 1 : -1;
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_dsv41::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> result;
    for (const auto & [ctx, buffer] : ctxs_bufs) {
        auto buft = ggml_backend_buffer_get_type(buffer.get());
        result[buft] += hparams.no_alloc ? ggml_backend_alloc_ctx_tensors_from_buft_size(ctx.get(), buft) : ggml_backend_buffer_get_size(buffer.get());
    }
    result[ggml_backend_cpu_buffer_type()] += tokens.capacity()*sizeof(llama_token);
    return result;
}

void llama_memory_dsv41::read_raw(int il, llama_pos pos, float * values) const {
    const auto * raw = layers[il].raw;
    GGML_ASSERT(raw && pos >= 0);
    std::vector<uint8_t> packed(raw->ne[0]);
    ggml_backend_tensor_get(raw, packed.data(), (pos % n_ring)*raw->nb[1], packed.size());
    dequantize_row_mxfp8_act(packed.data(), values, hparams.n_embd_head_k(il));
}

void llama_memory_dsv41::read_kv(int il, llama_pos pos, float * values) const {
    const auto source = hparams.dsv41_kv_source[il];
    const auto * kv = layers[source].kv;
    GGML_ASSERT(kv && pos >= 0 && pos < kv->ne[1]);
    std::vector<block_nvfp4> packed(hparams.n_embd_head_k(il)/64);
    ggml_backend_tensor_get(kv, packed.data(), pos*kv->nb[1], packed.size()*sizeof(block_nvfp4));
    dequantize_row_nvfp4(packed.data(), values, hparams.n_embd_head_k(il));
}

void llama_memory_dsv41::read_index(int il, llama_pos pos, float * values) const {
    const auto source = hparams.dsv41_kv_source[il];
    const auto * index = layers[source].index;
    GGML_ASSERT(index && pos >= 0 && pos < index->ne[1]);
    std::vector<block_mxfp4> packed(hparams.indexer_head_size/32);
    ggml_backend_tensor_get(index, packed.data(), pos*index->nb[1], packed.size()*sizeof(block_mxfp4));
    dequantize_row_mxfp4(packed.data(), values, hparams.indexer_head_size);
}

void llama_memory_dsv41::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    if (flags & ~(LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY | LLAMA_STATE_SEQ_FLAGS_ON_DEVICE)) {
        throw std::runtime_error("unsupported DeepSeek-V4.1 state flags");
    }
    const bool partial = flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
    GGML_ASSERT(seq_id == -1 || seq_id == 0);
    const uint32_t header[] = { 0xd5410003, n_ring, (uint32_t) layers.size(), hparams.n_embd_head_k(), hparams.indexer_head_size, (uint32_t) tokens.size(), ring_end, decoder_start, decoder_ready, hparams.n_embd, hparams.dsv4_hc_mult, partial };
    io.write(header, sizeof(header));
    for (uint32_t il = 0; il < layers.size(); ++il) {
        const int32_t layout[] = { hparams.dsv41_kv_source[il], (int32_t) hparams.dsv4_compress_ratios[il] };
        io.write(layout, sizeof(layout));
    }
    io.write(tokens.data(), tokens.size()*sizeof(llama_token));
    for (auto * tensor : {encoder_hidden, encoder_pre}) {
        io.write_tensor(tensor, 0, std::min<size_t>(tokens.size(), n_ring)*tensor->nb[1]);
    }
    for (uint32_t il = 0; il < layers.size(); ++il) {
        const auto & layer = layers[il];
        for (auto * tensor : {layer.raw, layer.kv, layer.index, layer.comp_kv, layer.comp_score}) {
            if (partial && (tensor == layer.kv || tensor == layer.index)) { continue; }
            if (tensor) {
                const size_t rows = tensor == layer.kv || tensor == layer.index ? tokens.size()/hparams.dsv4_compress_ratios[il] : std::min<size_t>(tokens.size(), n_ring);
                io.write_tensor(tensor, 0, rows*tensor->nb[1]);
            }
        }
    }
}

void llama_memory_dsv41::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    if (flags & ~(LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY | LLAMA_STATE_SEQ_FLAGS_ON_DEVICE)) {
        throw std::runtime_error("unsupported DeepSeek-V4.1 state flags");
    }
    const bool partial = flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
    GGML_ASSERT(seq_id == -1 || seq_id == 0);
    try {
        uint32_t header[12];
        io.read(header, sizeof(header));
        if (header[0] != 0xd5410003 || header[1] != n_ring || header[2] != layers.size() || header[3] != hparams.n_embd_head_k() || header[4] != hparams.indexer_head_size || header[5] > header[6] || header[6] > n_ctx || header[7] > header[5] || header[8] > 1 || header[9] != hparams.n_embd || header[10] != hparams.dsv4_hc_mult || header[11] != partial) {
            throw std::runtime_error("DeepSeek-V4.1 state layout mismatch");
        }
        for (uint32_t il = 0; il < layers.size(); ++il) {
            int32_t layout[2];
            io.read(layout, sizeof(layout));
            if (layout[0] != hparams.dsv41_kv_source[il] || layout[1] != (int32_t) hparams.dsv4_compress_ratios[il]) {
                throw std::runtime_error("DeepSeek-V4.1 state source layout mismatch");
            }
        }
        std::vector<llama_token> restored(header[5]);
        io.read(restored.data(), restored.size()*sizeof(llama_token));
        for (auto token : restored) {
            if (token < 0 || (uint32_t) token >= n_vocab) {
                throw std::runtime_error("invalid token in DeepSeek-V4.1 state");
            }
        }
        if (partial && (restored.size() > tokens.size() || !std::equal(restored.begin(), restored.end(), tokens.begin()))) {
            throw std::runtime_error("DeepSeek-V4.1 partial state requires the matching global context");
        }
        for (auto * tensor : {encoder_hidden, encoder_pre}) {
            io.read_tensor(tensor, 0, std::min<size_t>(restored.size(), n_ring)*tensor->nb[1]);
        }
        for (uint32_t il = 0; il < layers.size(); ++il) {
            const auto & layer = layers[il];
            for (auto * tensor : {layer.raw, layer.kv, layer.index, layer.comp_kv, layer.comp_score}) {
                if (partial && (tensor == layer.kv || tensor == layer.index)) { continue; }
                if (tensor) {
                    const size_t rows = tensor == layer.kv || tensor == layer.index ? restored.size()/hparams.dsv4_compress_ratios[il] : std::min<size_t>(restored.size(), n_ring);
                    io.read_tensor(tensor, 0, rows*tensor->nb[1]);
                }
            }
        }
        tokens = std::move(restored);
        ring_end = header[6];
        decoder_start = header[7];
        decoder_ready = header[8];
    } catch (...) {
        clear(false);
        throw;
    }
}

llama_memory_dsv41_context::llama_memory_dsv41_context(llama_memory_dsv41 * memory, std::vector<llama_ubatch> ubatches, llama_memory_status status) :
    memory(memory), ubatches(std::move(ubatches)), status(status) {}

bool llama_memory_dsv41_context::next() {
    return ++index < ubatches.size();
}

bool llama_memory_dsv41_context::apply() {
    return status == LLAMA_MEMORY_STATUS_SUCCESS && (ubatches.empty() || memory->apply(ubatches[index]));
}

const llama_ubatch & llama_memory_dsv41_context::get_ubatch() const {
    GGML_ASSERT(index < ubatches.size());
    return ubatches[index];
}

uint32_t llama_memory_dsv41_context::get_replay_tokens() const {
    return ubatches.empty() ? memory->hparams.n_swa : std::min<size_t>(memory->tokens.size(), memory->hparams.n_swa);
}
