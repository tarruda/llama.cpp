#include "llama-memory-qwen4.h"

#include "llama-context.h"
#include "llama-impl.h"
#include "llama-io.h"
#include "llama-model.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <tuple>

static bool qwen4_pos_in(llama_pos pos, llama_pos p0, llama_pos p1) {
    const llama_pos begin = p0 < 0 ? 0 : p0;
    const llama_pos end = p1 < 0 ? std::numeric_limits<llama_pos>::max() : p1;
    return pos >= begin && pos < end;
}

static uint32_t qwen4_qsa_ratio(const llama_hparams & hparams, bool mtp) {
    for (uint32_t il = 0; il < hparams.n_layer_all; ++il) {
        if ((mtp ? il >= hparams.n_layer() : il < hparams.n_layer()) && !hparams.is_recr(il)) {
            return hparams.qwen4_compress_ratios[il];
        }
    }

    GGML_ABORT("Qwen4 memory has no sparse-attention layer");
}

static llama_hparams qwen4_qsa_hparams(const llama_hparams & hparams) {
    llama_hparams result = hparams;
    std::fill(result.n_head_kv_arr.begin(), result.n_head_kv_arr.end(), 1);
    result.n_embd_head_k_full = hparams.indexer_head_size;
    result.n_embd_head_v_full = hparams.indexer_head_size;
    result.n_embd_head_k_swa = hparams.indexer_head_size;
    result.n_embd_head_v_swa = hparams.indexer_head_size;
    // llama_kv_cache uses MLA dimensions to allocate K-only storage.
    result.n_embd_head_k_mla_impl = hparams.indexer_head_size;
    result.n_embd_head_v_mla_impl = hparams.indexer_head_size;
    return result;
}

llama_memory_qwen4::llama_memory_qwen4(
        const llama_model & model,
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                 uint32_t   kv_size,
                 uint32_t   n_pad,
                 uint32_t   n_seq_max,
                 uint32_t   n_rs_seq,
                     bool   offload,
                     bool   unified,
                     bool   mtp) :
    hparams(model.hparams),
    hparams_qsa(qwen4_qsa_hparams(model.hparams)),
    mem_attn(new llama_kv_cache_msa(
        model,
        type_k,
        type_v,
        v_trans,
        offload,
        unified,
        kv_size,
        n_seq_max,
        n_pad,
        0,
        LLAMA_SWA_TYPE_NONE,
        [&](int32_t il) { return mtp ? il >= (int32_t) hparams.n_layer() : il < (int32_t) hparams.n_layer() && !hparams.is_recr(il); },
        [&](int32_t il) { return mtp ? il >= (int32_t) hparams.n_layer() : il < (int32_t) hparams.n_layer() && !hparams.is_recr(il); },
        nullptr)),
    mem_qsa(new llama_kv_cache(
        model,
        hparams_qsa,
        GGML_TYPE_F32,
        GGML_TYPE_F32,
        v_trans,
        offload,
        false,
        GGML_PAD((kv_size + qwen4_qsa_ratio(hparams, mtp) - 1)/qwen4_qsa_ratio(hparams, mtp) + 1, 256u),
        n_seq_max,
        n_pad,
        0,
        LLAMA_SWA_TYPE_NONE,
        nullptr,
        [&](int32_t il) { return mtp ? il >= (int32_t) hparams.n_layer() : il < (int32_t) hparams.n_layer() && !hparams.is_recr(il); },
        nullptr,
        nullptr)),
    mem_gdn(!mtp ? new llama_memory_recurrent(
        model,
        GGML_TYPE_F32,
        GGML_TYPE_F32,
        offload,
        std::max(1u, n_seq_max),
        n_seq_max,
        n_rs_seq,
        [&](int32_t il) { return il < (int32_t) hparams.n_layer() && hparams.is_recr(il); }) : nullptr),
    mem_ple(!mtp && hparams.qwen4_ple_layer >= 0 ? new llama_memory_recurrent(
        model,
        GGML_TYPE_F32,
        GGML_TYPE_F32,
        offload,
        std::max(1u, n_seq_max),
        n_seq_max,
        n_rs_seq,
        [&](int32_t il) { return il == hparams.qwen4_ple_layer; },
        hparams.n_embd * hparams.qwen4_hc_count * (hparams.qwen4_ple_conv - 1) * hparams.qwen4_ple_ngram,
        1,
        "ple_cache") : nullptr) {
}

llama_memory_context_ptr llama_memory_qwen4::init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) {
    do {
        balloc.split_reset();

        std::vector<llama_ubatch> ubatches;
        while (true) {
            llama_ubatch ubatch;
            if (embd_all) {
                ubatch = balloc.split_seq(n_ubatch);
            } else {
                const bool unified = mem_attn->get_base()->get_n_stream() == 1;
                const uint32_t n_rs_seq = mem_gdn ? mem_gdn->n_rs_seq : 0;
                ubatch = balloc.split_equal(n_ubatch, !unified, n_rs_seq > 0 ? n_rs_seq + 1 : 0);
            }

            if (ubatch.n_tokens == 0) {
                break;
            }
            ubatches.push_back(std::move(ubatch));
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            break;
        }

        if ((mem_gdn && !mem_gdn->prepare(ubatches)) || (mem_ple && !mem_ple->prepare(ubatches))) {
            LLAMA_LOG_ERROR("%s: failed to prepare recurrent ubatches\n", __func__);
            break;
        }

        auto sinfos_base = mem_attn->get_base()->prepare(ubatches);
        if (sinfos_base.empty()) {
            LLAMA_LOG_ERROR("%s: failed to prepare attention ubatches\n", __func__);
            break;
        }

        auto sinfos_idx = mem_attn->get_idx()->prepare(ubatches);
        if (sinfos_idx.empty()) {
            LLAMA_LOG_ERROR("%s: failed to prepare indexer ubatches\n", __func__);
            break;
        }

        return std::make_unique<llama_memory_qwen4_context>(
                this, std::move(sinfos_base), std::move(sinfos_idx), std::move(ubatches));
    } while (false);

    return std::make_unique<llama_memory_qwen4_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_memory_qwen4::init_full() {
    return std::make_unique<llama_memory_qwen4_context>(this);
}

llama_memory_context_ptr llama_memory_qwen4::init_update(llama_context * lctx, bool optimize) {
    return std::make_unique<llama_memory_qwen4_context>(this, lctx, optimize);
}

bool llama_memory_qwen4::get_can_shift() const {
    return mem_attn->get_can_shift();
}

void llama_memory_qwen4::clear(bool data) {
    mem_attn->clear(data);
    mem_qsa->clear(data);
    if (mem_gdn) {
        mem_gdn->clear(data);
    }
    if (mem_ple) {
        mem_ple->clear(data);
    }
    token_histories.clear();
    qsa_cached_blocks.clear();
}

bool llama_memory_qwen4::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    if ((mem_gdn && !mem_gdn->seq_rm(seq_id, p0, p1)) || (mem_ple && !mem_ple->seq_rm(seq_id, p0, p1))) {
        return false;
    }
    if (!mem_attn->seq_rm(seq_id, p0, p1)) {
        return false;
    }

    if (seq_id < 0) {
        qsa_cached_blocks.clear();
    } else {
        qsa_cached_blocks.erase(seq_id);
    }

    if (seq_id < 0) {
        for (auto & [_, history] : token_histories) {
            history.erase(std::remove_if(history.begin(), history.end(), [&](const token_entry & entry) {
                return qwen4_pos_in(entry.pos[0], p0, p1);
            }), history.end());
        }
    } else {
        auto it = token_histories.find(seq_id);
        if (it != token_histories.end()) {
            auto & history = it->second;
            history.erase(std::remove_if(history.begin(), history.end(), [&](const token_entry & entry) {
                return qwen4_pos_in(entry.pos[0], p0, p1);
            }), history.end());
            if (history.empty()) {
                token_histories.erase(it);
            }
        }
    }
    return true;
}

void llama_memory_qwen4::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    if (seq_id_src == seq_id_dst) {
        return;
    }

    qsa_cached_blocks.erase(seq_id_dst);

    token_history copied;
    const auto & cells_src = mem_attn->get_base()->get_cells(seq_id_src);
    const auto & cells_dst = mem_attn->get_base()->get_cells(seq_id_dst);
    const bool replace = &cells_src != &cells_dst;
    const auto src = token_histories.find(seq_id_src);
    if (src != token_histories.end()) {
        using pos_key = std::tuple<llama_pos, llama_pos, llama_pos>;
        std::map<pos_key, std::vector<bool>> cells_by_pos;
        for (uint32_t cell = 0; cell < cells_src.size(); ++cell) {
            if (cells_src.is_empty(cell) || !cells_src.seq_has(cell, seq_id_src) || !qwen4_pos_in(cells_src.pos_get(cell), p0, p1)) {
                continue;
            }
            const auto & ext = cells_src.ext_get(cell);
            cells_by_pos[{ cells_src.pos_get(cell), ext.y, ext.x }].push_back(!replace && cells_src.seq_has(cell, seq_id_dst));
        }

        std::map<pos_key, size_t> next_cell;
        for (const auto & entry : src->second) {
            if (!qwen4_pos_in(entry.pos[0], p0, p1)) {
                continue;
            }
            const pos_key key = { entry.pos[0], entry.pos[1], entry.pos[2] };
            auto cells = cells_by_pos.find(key);
            if (cells == cells_by_pos.end()) {
                continue;
            }
            size_t & index = next_cell[key];
            if (index < cells->second.size() && !cells->second[index++]) {
                copied.push_back(entry);
            }
        }
    }

    mem_attn->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    if (mem_gdn) {
        mem_gdn->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    }
    if (mem_ple) {
        mem_ple->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    }

    if (replace) {
        if (copied.empty()) {
            token_histories.erase(seq_id_dst);
        } else {
            token_histories[seq_id_dst] = std::move(copied);
        }
    } else if (!copied.empty()) {
        auto & dst = token_histories[seq_id_dst];
        dst.insert(dst.end(), copied.begin(), copied.end());
    }
}

void llama_memory_qwen4::seq_keep(llama_seq_id seq_id) {
    mem_attn->seq_keep(seq_id);
    if (mem_gdn) {
        mem_gdn->seq_keep(seq_id);
    }
    if (mem_ple) {
        mem_ple->seq_keep(seq_id);
    }

    const auto qsa_it = qsa_cached_blocks.find(seq_id);
    const uint32_t qsa_blocks = qsa_it == qsa_cached_blocks.end() ? 0 : qsa_it->second;
    qsa_cached_blocks.clear();
    if (qsa_blocks > 0) {
        qsa_cached_blocks.emplace(seq_id, qsa_blocks);
    }

    auto it = token_histories.find(seq_id);
    token_history keep = it == token_histories.end() ? token_history{} : std::move(it->second);
    token_histories.clear();
    if (!keep.empty()) {
        token_histories.emplace(seq_id, std::move(keep));
    }
}

void llama_memory_qwen4::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    mem_attn->seq_add(seq_id, p0, p1, shift);
    if (mem_gdn) {
        mem_gdn->seq_add(seq_id, p0, p1, shift);
    }
    if (mem_ple) {
        mem_ple->seq_add(seq_id, p0, p1, shift);
    }
    qsa_cached_blocks.erase(seq_id);

    auto it = token_histories.find(seq_id);
    if (it != token_histories.end()) {
        for (auto & entry : it->second) {
            if (qwen4_pos_in(entry.pos[0], p0, p1)) {
                entry.pos[0] += shift;
            }
        }
    }
}

void llama_memory_qwen4::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    mem_attn->seq_div(seq_id, p0, p1, d);
    if (mem_gdn) {
        mem_gdn->seq_div(seq_id, p0, p1, d);
    }
    if (mem_ple) {
        mem_ple->seq_div(seq_id, p0, p1, d);
    }
    qsa_cached_blocks.erase(seq_id);

    auto it = token_histories.find(seq_id);
    if (it != token_histories.end()) {
        for (auto & entry : it->second) {
            if (qwen4_pos_in(entry.pos[0], p0, p1)) {
                entry.pos[0] /= d;
            }
        }
    }
}

llama_pos llama_memory_qwen4::seq_pos_min(llama_seq_id seq_id) const {
    llama_pos result = mem_attn->seq_pos_min(seq_id);
    if (mem_gdn) {
        result = std::max(result, mem_gdn->seq_pos_min(seq_id));
    }
    if (mem_ple) {
        result = std::max(result, mem_ple->seq_pos_min(seq_id));
    }
    return result;
}

llama_pos llama_memory_qwen4::seq_pos_max(llama_seq_id seq_id) const {
    llama_pos result = mem_attn->seq_pos_max(seq_id);
    if (mem_gdn) {
        result = std::min(result, mem_gdn->seq_pos_max(seq_id));
    }
    if (mem_ple) {
        result = std::min(result, mem_ple->seq_pos_max(seq_id));
    }
    return result;
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_qwen4::memory_breakdown() const {
    auto result = mem_attn->memory_breakdown();
    for (const auto & [buft, size] : mem_qsa->memory_breakdown()) {
        result[buft] += size;
    }
    if (mem_gdn) {
        for (const auto & [buft, size] : mem_gdn->memory_breakdown()) {
            result[buft] += size;
        }
    }
    if (mem_ple) {
        for (const auto & [buft, size] : mem_ple->memory_breakdown()) {
            result[buft] += size;
        }
    }
    return result;
}

void llama_memory_qwen4::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        mem_attn->state_write(io, seq_id, flags);
    }
    if (mem_gdn) {
        mem_gdn->state_write(io, seq_id, flags);
    }
    if (mem_ple) {
        mem_ple->state_write(io, seq_id, flags);
    }

    uint32_t n_histories = 0;
    if (seq_id < 0) {
        n_histories = (uint32_t) token_histories.size();
    } else if (token_histories.count(seq_id) != 0) {
        n_histories = 1;
    }
    io.write(&n_histories, sizeof(n_histories));

    for (const auto & [stored_seq, history] : token_histories) {
        if (seq_id >= 0 && stored_seq != seq_id) {
            continue;
        }
        io.write(&stored_seq, sizeof(stored_seq));
        const uint64_t n_entries = history.size();
        io.write(&n_entries, sizeof(n_entries));
        for (const auto & entry : history) {
            io.write(entry.pos.data(), sizeof(entry.pos));
            io.write(&entry.token, sizeof(entry.token));
        }
    }
}

void llama_memory_qwen4::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        mem_attn->state_read(io, seq_id, flags);
    }
    if (mem_gdn) {
        mem_gdn->state_read(io, seq_id, flags);
    }
    if (mem_ple) {
        mem_ple->state_read(io, seq_id, flags);
    }

    uint32_t n_histories;
    io.read(&n_histories, sizeof(n_histories));
    if (n_histories > LLAMA_MAX_SEQ) {
        throw std::runtime_error("invalid Qwen4 token history count");
    }

    if (seq_id < 0) {
        token_histories.clear();
    } else {
        token_histories.erase(seq_id);
    }

    for (uint32_t ih = 0; ih < n_histories; ++ih) {
        llama_seq_id stored_seq;
        uint64_t n_entries;
        io.read(&stored_seq, sizeof(stored_seq));
        io.read(&n_entries, sizeof(n_entries));
        if (n_entries > std::numeric_limits<uint32_t>::max()) {
            throw std::runtime_error("invalid Qwen4 token history length");
        }

        auto & history = token_histories[seq_id < 0 ? stored_seq : seq_id];
        history.resize(n_entries);
        for (auto & entry : history) {
            io.read(entry.pos.data(), sizeof(entry.pos));
            io.read(&entry.token, sizeof(entry.token));
        }
    }

    if (seq_id < 0) {
        qsa_cached_blocks.clear();
    } else {
        qsa_cached_blocks.erase(seq_id);
    }
}

llama_kv_cache_msa * llama_memory_qwen4::get_mem_attn() const {
    return mem_attn.get();
}

llama_kv_cache * llama_memory_qwen4::get_mem_qsa() const {
    return mem_qsa.get();
}

llama_memory_recurrent * llama_memory_qwen4::get_mem_gdn() const {
    return mem_gdn.get();
}

llama_memory_recurrent * llama_memory_qwen4::get_mem_ple() const {
    return mem_ple.get();
}

bool llama_memory_qwen4::prepare_ple_ids(const llama_ubatch & ubatch, std::vector<int32_t> & ids) const {
    const llama_token * tokens = ubatch.token_orig ? ubatch.token_orig : ubatch.token;
    if (!tokens) {
        LLAMA_LOG_ERROR("%s: Qwen4 PLE requires original token IDs for embedding inputs\n", __func__);
        return false;
    }

    const uint32_t ngram = hparams.qwen4_ple_ngram;
    const uint32_t heads_per_ngram = hparams.qwen4_ple_heads;
    const uint32_t n_heads = (ngram - 1) * heads_per_ngram;
    const llama_token eos = (llama_token) hparams.qwen4_ple_eos;
    ids.resize((size_t) n_heads * ubatch.n_tokens);

    std::map<llama_seq_id, std::vector<llama_token>> tails;
    auto get_tail = [&](llama_seq_id seq_id) -> std::vector<llama_token> & {
        auto [it, inserted] = tails.emplace(seq_id, std::vector<llama_token>{});
        if (inserted) {
            const auto found = token_histories.find(seq_id);
            if (found != token_histories.end()) {
                const auto & history = found->second;
                const size_t start = history.size() > ngram - 1 ? history.size() - (ngram - 1) : 0;
                for (size_t i = start; i < history.size(); ++i) {
                    it->second.push_back(history[i].token);
                }
            }
        }
        return it->second;
    };

    const auto hash_token = [&](const std::vector<llama_token> & tail, llama_token token, uint32_t order) {
        uint64_t mixed = uint64_t(int64_t(token)) * uint64_t(hparams.qwen4_ple_multipliers[0]);
        for (uint32_t ip = 1; ip < order; ++ip) {
            llama_token previous = eos;
            if (tail.size() >= ip) {
                previous = tail[tail.size() - ip];
                for (size_t j = tail.size() - ip; j < tail.size(); ++j) {
                    if (tail[j] == eos) {
                        previous = eos;
                        break;
                    }
                }
            }
            mixed ^= uint64_t(int64_t(previous)) * uint64_t(hparams.qwen4_ple_multipliers[ip]);
        }
        return int64_t(mixed);
    };

    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        const llama_token token = tokens[i];
        const llama_seq_id seq_id = ubatch.seq_id[i][0];
        auto & tail = get_tail(seq_id);

        for (uint32_t order = 2; order <= ngram; ++order) {
            const int64_t mixed = hash_token(tail, token, order);
            const uint32_t head_start = (order - 2) * heads_per_ngram;
            for (uint32_t ih = 0; ih < heads_per_ngram; ++ih) {
                const uint32_t head = head_start + ih;
                const int64_t vocab = hparams.qwen4_ple_vocab_sizes[head];
                int64_t row = mixed % vocab;
                if (row < 0) {
                    row += vocab;
                }
                row += hparams.qwen4_ple_offsets[head];
                if (row > std::numeric_limits<int32_t>::max()) {
                    LLAMA_LOG_ERROR("%s: Qwen4 PLE row index exceeds I32\n", __func__);
                    return false;
                }
                ids[(size_t) i * n_heads + head] = row;
            }
        }

        tail.push_back(token);
        if (tail.size() > ngram - 1) {
            tail.erase(tail.begin());
        }

        for (int32_t is = 1; is < ubatch.n_seq_id[i]; ++is) {
            auto & other = get_tail(ubatch.seq_id[i][is]);
            other.push_back(token);
            if (other.size() > ngram - 1) {
                other.erase(other.begin());
            }
        }
    }
    return true;
}

void llama_memory_qwen4::commit_tokens(const llama_ubatch & ubatch) {
    const llama_token * tokens = ubatch.token_orig ? ubatch.token_orig : ubatch.token;
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        token_entry entry = { {}, tokens ? tokens[i] : LLAMA_TOKEN_NULL };
        for (uint32_t ip = 0; ip < entry.pos.size(); ++ip) {
            entry.pos[ip] = ip < ubatch.n_pos ? ubatch.pos[i + ip * ubatch.n_tokens] : ubatch.pos[i];
        }
        for (int32_t is = 0; is < ubatch.n_seq_id[i]; ++is) {
            token_histories[ubatch.seq_id[i][is]].push_back(entry);
        }
    }
}

uint32_t llama_memory_qwen4::get_qsa_update_capacity(
        const llama_ubatch & ubatch,
        uint32_t ratio,
        uint32_t n_blocks,
        bool reserve) const {
    if (reserve) {
        return n_blocks * std::max(1u, ubatch.n_seqs_unq);
    }

    std::map<llama_seq_id, bool> active;
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        for (int32_t is = 0; is < ubatch.n_seq_id[i]; ++is) {
            active.emplace(ubatch.seq_id[i][is], true);
        }
    }

    uint32_t n_missing = 0;
    for (const auto & [seq_id, _] : active) {
        const auto history = token_histories.find(seq_id);
        const uint32_t n_complete = history == token_histories.end() ? 0 : history->second.size()/ratio;
        const auto cached = qsa_cached_blocks.find(seq_id);
        const uint32_t n_cached = cached == qsa_cached_blocks.end() || cached->second > n_complete ? 0 : cached->second;
        n_missing += n_complete - n_cached;
    }

    const uint32_t n_normal = ubatch.n_tokens/ratio + std::max(1u, ubatch.n_seqs_unq);
    return std::max(n_normal, n_missing);
}

void llama_memory_qwen4::set_input_qsa_layout(
        ggml_tensor * block_cells,
        ggml_tensor * block_key_cells,
        ggml_tensor * block_mask,
        ggml_tensor * selected,
        ggml_tensor * update_cells,
        ggml_tensor * update_pos,
        ggml_tensor * update_idxs,
        const ggml_tensor * kq_mask,
        const llama_ubatch * ubatch,
        uint32_t ratio,
        uint32_t block_topk,
        uint32_t n_kv,
        uint32_t n_stream) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(block_cells->buffer));
    GGML_ASSERT(ggml_backend_buffer_is_host(block_key_cells->buffer));
    GGML_ASSERT(ggml_backend_buffer_is_host(block_mask->buffer));
    GGML_ASSERT(ggml_backend_buffer_is_host(selected->buffer));
    GGML_ASSERT(ggml_backend_buffer_is_host(update_cells->buffer));
    GGML_ASSERT(ggml_backend_buffer_is_host(update_pos->buffer));
    GGML_ASSERT(ggml_backend_buffer_is_host(update_idxs->buffer));
    GGML_ASSERT(ggml_backend_buffer_is_host(kq_mask->buffer));
    GGML_ASSERT(block_cells->type == GGML_TYPE_I32);
    GGML_ASSERT(block_key_cells->type == GGML_TYPE_I32);
    GGML_ASSERT(block_mask->type == GGML_TYPE_F32);
    GGML_ASSERT(selected->type == GGML_TYPE_F32);
    GGML_ASSERT(update_cells->type == GGML_TYPE_I32);
    GGML_ASSERT(update_pos->type == GGML_TYPE_I32);
    GGML_ASSERT(update_idxs->type == GGML_TYPE_I32);

    const int64_t n_blocks = block_cells->ne[1];
    const int64_t n_tokens = ubatch->n_tokens;
    const int64_t n_pos = update_pos->ne[1];
    const int64_t n_updates = update_cells->ne[1];
    const int64_t n_tps = n_tokens/n_stream;
    const int64_t qsa_size = mem_qsa->get_size();
    const int64_t qsa_streams = mem_qsa->get_n_stream();
    const int64_t raw_size = mem_attn->get_idx()->get_size();
    const int64_t raw_streams = mem_attn->get_idx()->get_n_stream();

    GGML_ASSERT(block_cells->ne[0] == ratio && block_cells->ne[2] == n_tokens);
    GGML_ASSERT(block_key_cells->ne[0] == n_blocks && block_key_cells->ne[1] == n_tokens);
    GGML_ASSERT(block_mask->ne[0] == n_blocks && block_mask->ne[1] == n_tokens);
    GGML_ASSERT(update_cells->ne[0] == ratio);
    GGML_ASSERT(update_pos->ne[0] == n_updates);
    GGML_ASSERT(ggml_nelements(update_idxs) == n_updates);
    GGML_ASSERT(n_stream > 0 && n_tokens % n_stream == 0);
    GGML_ASSERT(selected->ne[1] == n_tokens);
    GGML_ASSERT(selected->ne[0] == n_kv);
    GGML_ASSERT(kq_mask->ne[0] == n_kv);

    auto * cell_data = (int32_t *) block_cells->data;
    auto * key_cell_data = (int32_t *) block_key_cells->data;
    auto * mask_data = (float *) block_mask->data;
    auto * selected_data = (float *) selected->data;
    auto * update_cell_data = (int32_t *) update_cells->data;
    auto * update_pos_data = (int32_t *) update_pos->data;
    auto * update_idx_data = (int32_t *) update_idxs->data;

    std::fill(cell_data, cell_data + ggml_nelements(block_cells), 0);
    std::fill(key_cell_data, key_cell_data + ggml_nelements(block_key_cells), 0);
    std::fill(mask_data, mask_data + ggml_nelements(block_mask), -INFINITY);
    std::fill(selected_data, selected_data + ggml_nelements(selected), 0.0f);
    std::fill(update_cell_data, update_cell_data + ggml_nelements(update_cells), 0);
    std::fill(update_pos_data, update_pos_data + ggml_nelements(update_pos), 0);
    std::fill(update_idx_data, update_idx_data + ggml_nelements(update_idxs), 0);

    const auto mask_visible = [&](int64_t query, uint32_t cell) {
        const int64_t index = query * n_kv + cell;
        if (kq_mask->type == GGML_TYPE_F16) {
            return std::isfinite(ggml_fp16_to_fp32(((const ggml_fp16_t *) kq_mask->data)[index]));
        }
        return std::isfinite(((const float *) kq_mask->data)[index]);
    };

    using mapped_token = std::pair<const token_entry *, uint32_t>;
    std::map<llama_seq_id, std::vector<mapped_token>> mapped_by_seq;
    std::map<llama_seq_id, uint32_t> stream_by_seq;

    for (int64_t iq = 0; iq < n_tokens; ++iq) {
        const uint32_t stream = iq/n_tps;
        for (int32_t is = 0; is < ubatch->n_seq_id[iq]; ++is) {
            const llama_seq_id seq_id = ubatch->seq_id[iq][is];
            const auto [it, inserted] = stream_by_seq.emplace(seq_id, stream);
            GGML_ASSERT(inserted || it->second == stream);
        }
    }

    for (const auto & [seq_id, _] : stream_by_seq) {
        const auto found = token_histories.find(seq_id);
        if (found == token_histories.end()) {
            continue;
        }

        const auto & cells = mem_attn->get_base()->get_cells(seq_id);
        using pos_key = std::tuple<llama_pos, llama_pos, llama_pos>;
        std::map<pos_key, std::vector<uint32_t>> cells_by_pos;
        for (uint32_t cell = 0; cell < cells.size() && cell < (uint32_t) n_kv; ++cell) {
            if (cells.is_empty(cell) || !cells.seq_has(cell, seq_id)) {
                continue;
            }
            const auto & ext = cells.ext_get(cell);
            cells_by_pos[{ cells.pos_get(cell), ext.y, ext.x }].push_back(cell);
        }

        std::map<pos_key, size_t> next_cell;
        auto & mapped = mapped_by_seq[seq_id];
        for (const auto & entry : found->second) {
            const pos_key key = { entry.pos[0], entry.pos[1], entry.pos[2] };
            auto cells_it = cells_by_pos.find(key);
            if (cells_it == cells_by_pos.end()) {
                continue;
            }
            size_t & index = next_cell[key];
            if (index >= cells_it->second.size()) {
                continue;
            }
            const uint32_t cell = cells_it->second[index++];
            mapped.emplace_back(&entry, cell);
        }
    }

    std::map<llama_seq_id, std::vector<int32_t>> block_by_first_cell;
    for (const auto & [seq_id, mapped] : mapped_by_seq) {
        auto & blocks = block_by_first_cell[seq_id];
        blocks.resize(n_kv, -1);
        for (size_t ib = 0; ib < mapped.size()/ratio; ++ib) {
            GGML_ASSERT(ib <= (size_t) std::numeric_limits<int32_t>::max());
            blocks[mapped[ib*ratio].second] = (int32_t) ib;
        }
    }

    int64_t n_update = 0;
    llama_seq_id dummy_seq = -1;
    const mapped_token * dummy_token = nullptr;

    for (const auto & [seq_id, mapped] : mapped_by_seq) {
        GGML_ASSERT(seq_id >= 0 && seq_id < qsa_streams);
        const uint32_t n_complete = mapped.size()/ratio;
        const auto cached = qsa_cached_blocks.find(seq_id);
        const uint32_t n_cached = cached == qsa_cached_blocks.end() || cached->second > n_complete ? 0 : cached->second;
        const int64_t raw_stream = raw_streams == 1 ? 0 : seq_id;
        GGML_ASSERT(raw_stream >= 0 && raw_stream < raw_streams);

        if (!mapped.empty() && !dummy_token) {
            dummy_seq = seq_id;
            dummy_token = &mapped[0];
        }

        for (uint32_t ib = n_cached; ib < n_complete; ++ib) {
            GGML_ASSERT(n_update < n_updates);
            for (uint32_t ir = 0; ir < ratio; ++ir) {
                const int64_t source = raw_stream*raw_size + mapped[(size_t) ib*ratio + ir].second;
                GGML_ASSERT(source <= std::numeric_limits<int32_t>::max());
                update_cell_data[n_update*ratio + ir] = source;
            }
            for (int64_t ip = 0; ip < n_pos; ++ip) {
                update_pos_data[ip*n_updates + n_update] = mapped[(size_t) ib*ratio].first->pos[ip];
            }
            const int64_t dst = (int64_t) seq_id*qsa_size + ib;
            GGML_ASSERT(dst <= std::numeric_limits<int32_t>::max());
            update_idx_data[n_update++] = dst;
        }
        qsa_cached_blocks[seq_id] = n_complete;
    }

    GGML_ASSERT(dummy_token);
    const int64_t dummy_raw_stream = raw_streams == 1 ? 0 : dummy_seq;
    const int64_t dummy_source = dummy_raw_stream*raw_size + dummy_token->second;
    const int64_t dummy_dst = (int64_t) dummy_seq*qsa_size + qsa_size - 1;
    GGML_ASSERT(dummy_source <= std::numeric_limits<int32_t>::max());
    GGML_ASSERT(dummy_dst <= std::numeric_limits<int32_t>::max());
    while (n_update < n_updates) {
        for (uint32_t ir = 0; ir < ratio; ++ir) {
            update_cell_data[n_update*ratio + ir] = dummy_source;
        }
        for (int64_t ip = 0; ip < n_pos; ++ip) {
            update_pos_data[ip*n_updates + n_update] = dummy_token->first->pos[ip];
        }
        update_idx_data[n_update++] = dummy_dst;
    }

    for (int64_t iq = 0; iq < n_tokens; ++iq) {
        const llama_seq_id seq_id = ubatch->seq_id[iq][0];
        const auto mapped_it = mapped_by_seq.find(seq_id);
        if (mapped_it == mapped_by_seq.end()) {
            continue;
        }
        const auto & mapped = mapped_it->second;
        const auto & blocks = block_by_first_cell.at(seq_id);
        const int64_t scratch = (int64_t) seq_id*qsa_size + qsa_size - 1;
        GGML_ASSERT(scratch <= std::numeric_limits<int32_t>::max());
        std::fill(key_cell_data + iq*n_blocks, key_cell_data + (iq + 1)*n_blocks, scratch);

        std::vector<mapped_token> visible;
        visible.reserve(mapped.size());
        for (const auto & item : mapped) {
            if (mask_visible(iq, item.second)) {
                visible.push_back(item);
            }
        }

        const size_t n_complete = visible.size() / ratio;
        const size_t n_write = std::min<size_t>(n_complete, n_blocks);
        std::vector<uint8_t> used_cells(n_kv, 0);
        for (size_t ib = 0; ib < n_write; ++ib) {
            mask_data[iq * n_blocks + ib] = 0.0f;
            const uint32_t first_cell = visible[ib*ratio].second;
            const int32_t full_block = blocks[first_cell];
            GGML_ASSERT(full_block >= 0 && (size_t) full_block < mapped.size()/ratio);
            for (uint32_t ir = 0; ir < ratio; ++ir) {
                const uint32_t cell = visible[ib * ratio + ir].second;
                GGML_ASSERT(mapped[(size_t) full_block*ratio + ir].second == cell);
                cell_data[(iq * n_blocks + ib) * ratio + ir] = cell;
                used_cells[cell] = 1;
            }
            const int64_t key_cell = (int64_t) seq_id*qsa_size + full_block;
            GGML_ASSERT(key_cell <= std::numeric_limits<int32_t>::max());
            key_cell_data[iq*n_blocks + ib] = key_cell;
        }

        uint32_t fallback_cell = 0;
        for (size_t ib = n_write; ib < (size_t) n_blocks; ++ib) {
            for (uint32_t ir = 0; ir < ratio; ++ir) {
                while (used_cells[fallback_cell]) {
                    ++fallback_cell;
                }
                cell_data[(iq * n_blocks + ib) * ratio + ir] = fallback_cell;
                used_cells[fallback_cell++] = 1;
            }
        }

        const size_t selected_start = n_complete <= block_topk ? 0 : n_complete * ratio;
        for (size_t iv = selected_start; iv < visible.size(); ++iv) {
            selected_data[iq * n_kv + visible[iv].second] = 1.0f;
        }
    }
}

llama_memory_qwen4_context::llama_memory_qwen4_context(llama_memory_status status) : status(status) {
}

llama_memory_qwen4_context::llama_memory_qwen4_context(llama_memory_qwen4 * mem) :
    mem(mem),
    ctx_attn(mem->get_mem_attn()->init_full()),
    ctx_gdn(mem->get_mem_gdn() ? mem->get_mem_gdn()->init_full() : nullptr),
    ctx_ple(mem->get_mem_ple() ? mem->get_mem_ple()->init_full() : nullptr),
    reserve(true),
    status(llama_memory_status_combine(
            ctx_attn->get_status(),
            llama_memory_status_combine(
                ctx_gdn ? ctx_gdn->get_status() : LLAMA_MEMORY_STATUS_SUCCESS,
                ctx_ple ? ctx_ple->get_status() : LLAMA_MEMORY_STATUS_SUCCESS))) {
}

llama_memory_qwen4_context::llama_memory_qwen4_context(llama_memory_qwen4 * mem, llama_context * lctx, bool optimize) :
    mem(mem),
    ctx_attn(mem->get_mem_attn()->init_update(lctx, optimize)),
    ctx_gdn(mem->get_mem_gdn() ? mem->get_mem_gdn()->init_update(lctx, optimize) : nullptr),
    ctx_ple(mem->get_mem_ple() ? mem->get_mem_ple()->init_update(lctx, optimize) : nullptr),
    status(llama_memory_status_combine(
            ctx_attn->get_status(),
            llama_memory_status_combine(
                ctx_gdn ? ctx_gdn->get_status() : LLAMA_MEMORY_STATUS_SUCCESS,
                ctx_ple ? ctx_ple->get_status() : LLAMA_MEMORY_STATUS_SUCCESS))) {
}

llama_memory_qwen4_context::llama_memory_qwen4_context(
        llama_memory_qwen4 * mem,
        slot_info_vec_t sinfos_base,
        slot_info_vec_t sinfos_idx,
        std::vector<llama_ubatch> ubatches) :
    mem(mem),
    ubatches(std::move(ubatches)),
    ctx_attn(new llama_kv_cache_msa_context(mem->get_mem_attn(), std::move(sinfos_base), std::move(sinfos_idx), this->ubatches)),
    ctx_gdn(mem->get_mem_gdn() ? new llama_memory_recurrent_context(mem->get_mem_gdn(), this->ubatches) : nullptr),
    ctx_ple(mem->get_mem_ple() ? new llama_memory_recurrent_context(mem->get_mem_ple(), this->ubatches) : nullptr),
    status(llama_memory_status_combine(
            ctx_attn->get_status(),
            llama_memory_status_combine(
                ctx_gdn ? ctx_gdn->get_status() : LLAMA_MEMORY_STATUS_SUCCESS,
                ctx_ple ? ctx_ple->get_status() : LLAMA_MEMORY_STATUS_SUCCESS))) {
}

bool llama_memory_qwen4_context::next() {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    ctx_attn->next();
    if (ctx_gdn) {
        ctx_gdn->next();
    }
    if (ctx_ple) {
        ctx_ple->next();
    }
    ple_ids.clear();

    return ++i_next < ubatches.size();
}

bool llama_memory_qwen4_context::apply() {
    assert(!llama_memory_status_is_fail(status));

    if (ubatches.empty()) {
        bool result = true;
        result = result & ctx_attn->apply();
        if (ctx_gdn) {
            result = result & ctx_gdn->apply();
        }
        if (ctx_ple) {
            result = result & ctx_ple->apply();
        }
        return result;
    }

    if (ctx_ple && !mem->prepare_ple_ids(ubatches[i_next], ple_ids)) {
        return false;
    }

    bool result = true;
    result = result & ctx_attn->apply();
    if (ctx_gdn) {
        result = result & ctx_gdn->apply();
    }
    if (ctx_ple) {
        result = result & ctx_ple->apply();
    }
    if (result) {
        mem->commit_tokens(ubatches[i_next]);
    }
    return result;
}

llama_memory_status llama_memory_qwen4_context::get_status() const {
    return status;
}

const llama_ubatch & llama_memory_qwen4_context::get_ubatch() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);
    return ubatches[i_next];
}

const llama_kv_cache_msa_context * llama_memory_qwen4_context::get_attn() const {
    return static_cast<const llama_kv_cache_msa_context *>(ctx_attn.get());
}

llama_kv_cache_msa * llama_memory_qwen4_context::get_attn_memory() const {
    return mem->get_mem_attn();
}

llama_kv_cache * llama_memory_qwen4_context::get_qsa() const {
    return mem->get_mem_qsa();
}

const llama_memory_recurrent_context * llama_memory_qwen4_context::get_gdn() const {
    return static_cast<const llama_memory_recurrent_context *>(ctx_gdn.get());
}

const llama_memory_recurrent_context * llama_memory_qwen4_context::get_ple() const {
    return static_cast<const llama_memory_recurrent_context *>(ctx_ple.get());
}

void llama_memory_qwen4_context::set_input_ple_ids(ggml_tensor * dst) const {
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    GGML_ASSERT(ggml_nelements(dst) == (int64_t) ple_ids.size());
    ggml_backend_tensor_set(dst, ple_ids.data(), 0, ple_ids.size() * sizeof(int32_t));
}

void llama_memory_qwen4_context::set_input_qsa_layout(
        ggml_tensor * block_cells,
        ggml_tensor * block_key_cells,
        ggml_tensor * block_mask,
        ggml_tensor * selected,
        ggml_tensor * update_cells,
        ggml_tensor * update_pos,
        ggml_tensor * update_idxs,
        const ggml_tensor * kq_mask,
        const llama_ubatch * ubatch,
        uint32_t ratio,
        uint32_t block_topk,
        uint32_t n_kv,
        uint32_t n_stream) const {
    mem->set_input_qsa_layout(
            block_cells, block_key_cells, block_mask, selected,
            update_cells, update_pos, update_idxs,
            kq_mask, ubatch, ratio, block_topk, n_kv, n_stream);
}

uint32_t llama_memory_qwen4_context::get_qsa_update_capacity(
        const llama_ubatch & ubatch,
        uint32_t ratio,
        uint32_t n_blocks) const {
    return mem->get_qsa_update_capacity(ubatch, ratio, n_blocks, reserve);
}
