#include "llama-memory-hybrid-idx.h"

#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-io.h"
#include "llama-model.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <tuple>

//
// llama_memory_hybrid_idx
//

llama_memory_hybrid_idx::llama_memory_hybrid_idx(
        const llama_model & model,
                            /* attn */
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                 uint32_t   kv_size,
                 uint32_t   n_pad,
                 uint32_t   n_swa,
           llama_swa_type   swa_type,
                            /* recurrent */
                ggml_type   type_r,
                ggml_type   type_s,
                 uint32_t   rs_size,
                            /* common */
                 uint32_t   n_seq_max,
                 uint32_t   n_rs_seq,
                     bool   offload,
                     bool   unified,
                            /* layer filters */
    const layer_filter_cb & filter_attn,
    const layer_filter_cb & filter_recr,
    const layer_filter_cb & filter_idx) :
    llama_memory_hybrid(
        model,
        type_k, type_v, v_trans, kv_size, n_pad, n_swa, swa_type,
        type_r, type_s, rs_size,
        n_seq_max, n_rs_seq, offload, unified,
        filter_attn, filter_recr),
    hparams_idx(model.hparams),
    mem_idx(filter_idx == nullptr ? nullptr : [&] {
        // MQA with a single key head of indexer_head_size, as llama_kv_cache_dsa shapes its own
        hparams_idx.rope_type = LLAMA_ROPE_TYPE_NONE;
        std::fill(hparams_idx.n_head_kv_arr.begin(), hparams_idx.n_head_kv_arr.end(), 1);
        hparams_idx.n_embd_head_k_full = model.hparams.indexer_head_size;

        LLAMA_LOG_INFO("%s: creating indexer KV cache, size = %u cells\n", __func__, kv_size);

        return new llama_kv_cache(
            model, hparams_idx, type_k, type_v, v_trans, offload, unified,
            kv_size, n_seq_max, n_pad, n_swa, swa_type,
            nullptr, filter_idx, nullptr, nullptr, "idx_");
    }()) {}

llama_memory_context_ptr llama_memory_hybrid_idx::init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) {
    // note: repeats llama_memory_hybrid::init_batch, as the indexer needs the attention slot infos that the base context hides
    do {
        balloc.split_reset();

        // follow the recurrent pattern for creating the ubatch splits
        std::vector<llama_ubatch> ubatches;

        while (true) {
            llama_ubatch ubatch;

            if (embd_all) {
                // if all tokens are output, split by sequence
                ubatch = balloc.split_seq(n_ubatch);
            } else {
                // Use non-sequential split when KV cache is unified (needed for hellaswag/winogrande/multiple-choice)
                const bool unified = (get_mem_attn()->get_n_stream() == 1);

                // [TAG_RECURRENT_ROLLBACK_SPLITS]
                // the trailing (1 + n_rs_seq) tokens of each seq must stay in the same ubatch
                //   so that the rollback snapshots remain valid
                const uint32_t n_rs_seq = get_mem_recr()->n_rs_seq;

                ubatch = balloc.split_equal(n_ubatch, !unified, n_rs_seq > 0 ? n_rs_seq + 1 : 0);
            }

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        // prepare the recurrent batches first
        if (!get_mem_recr()->prepare(ubatches)) {
            // TODO: will the recurrent cache be in an undefined context at this point?
            LLAMA_LOG_ERROR("%s: failed to prepare recurrent ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        // prepare the attention cache
        auto heads_attn = get_mem_attn()->prepare(ubatches);
        if (heads_attn.empty()) {
            LLAMA_LOG_ERROR("%s: failed to prepare attention ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        // the indexer uses the attention cache's slot layout; a separate one can drift from it
        llama_kv_cache::slot_info_vec_t heads_idx;
        if (mem_idx) {
            heads_idx = heads_attn;
        }

        return std::make_unique<llama_memory_hybrid_idx_context>(
                this, std::move(heads_attn), std::move(heads_idx), std::move(ubatches));
    } while(false);

    return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_full() {
    return std::make_unique<llama_memory_hybrid_idx_context>(this);
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_update(llama_context * lctx, bool optimize) {
    return std::make_unique<llama_memory_hybrid_idx_context>(this, lctx, optimize);
}

void llama_memory_hybrid_idx::clear(bool data) {
    llama_memory_hybrid::clear(data);

    if (mem_idx) {
        mem_idx->clear(data);
    }

    qsa_histories.clear();
}

bool llama_memory_hybrid_idx::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    // same order as llama_memory_hybrid::seq_rm: the recurrent cache can refuse, so try it first
    if (!get_mem_recr()->seq_rm(seq_id, p0, p1)) {
        return false;
    }

    if (mem_idx) {
        mem_idx->seq_rm(seq_id, p0, p1);
    }

    const bool res = get_mem_attn()->seq_rm(seq_id, p0, p1);
    if (!res) {
        return false;
    }

    auto remove = [&](qsa_history & history) {
        history.erase(std::remove_if(history.begin(), history.end(), [&](const qsa_token & token) {
            return (p0 < 0 || token.pos[0] >= p0) && (p1 < 0 || token.pos[0] < p1);
        }), history.end());
    };

    if (seq_id < 0) {
        for (auto & item : qsa_histories) {
            remove(item.second);
        }
    } else {
        auto it = qsa_histories.find(seq_id);
        if (it != qsa_histories.end()) {
            remove(it->second);
            if (it->second.empty()) {
                qsa_histories.erase(it);
            }
        }
    }

    return true;
}

void llama_memory_hybrid_idx::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    if (seq_id_src == seq_id_dst) {
        return;
    }

    qsa_history copied;
    const auto & cells_src = get_mem_attn()->get_cells(seq_id_src);
    const auto & cells_dst = get_mem_attn()->get_cells(seq_id_dst);
    const bool replace = &cells_src != &cells_dst;
    const auto src = qsa_histories.find(seq_id_src);
    if (src != qsa_histories.end()) {
        using pos_key = std::tuple<llama_pos, llama_pos, llama_pos>;
        std::map<pos_key, std::vector<bool>> cells_by_pos;
        for (uint32_t cell = 0; cell < cells_src.size(); ++cell) {
            if (cells_src.is_empty(cell) || !cells_src.seq_has(cell, seq_id_src)) {
                continue;
            }

            const llama_pos pos = cells_src.pos_get(cell);
            if ((p0 >= 0 && pos < p0) || (p1 >= 0 && pos >= p1)) {
                continue;
            }

            const auto & ext = cells_src.ext_get(cell);
            cells_by_pos[{ pos, ext.y, ext.x }].push_back(!replace && cells_src.seq_has(cell, seq_id_dst));
        }

        std::map<pos_key, size_t> next_cell;
        for (const auto & token : src->second) {
            if ((p0 >= 0 && token.pos[0] < p0) || (p1 >= 0 && token.pos[0] >= p1)) {
                continue;
            }

            const pos_key key = { token.pos[0], token.pos[1], token.pos[2] };
            auto cells = cells_by_pos.find(key);
            if (cells == cells_by_pos.end()) {
                continue;
            }

            size_t & index = next_cell[key];
            if (index < cells->second.size() && !cells->second[index++]) {
                copied.push_back(token);
            }
        }
    }

    llama_memory_hybrid::seq_cp(seq_id_src, seq_id_dst, p0, p1);

    if (mem_idx) {
        mem_idx->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    }

    if (replace) {
        if (copied.empty()) {
            qsa_histories.erase(seq_id_dst);
        } else {
            qsa_histories[seq_id_dst] = std::move(copied);
        }
    } else if (!copied.empty()) {
        auto & dst = qsa_histories[seq_id_dst];
        dst.insert(dst.end(), copied.begin(), copied.end());
    }
}

void llama_memory_hybrid_idx::seq_keep(llama_seq_id seq_id) {
    llama_memory_hybrid::seq_keep(seq_id);

    if (mem_idx) {
        mem_idx->seq_keep(seq_id);
    }

    auto it = qsa_histories.find(seq_id);
    qsa_history keep = it == qsa_histories.end() ? qsa_history{} : std::move(it->second);
    qsa_histories.clear();
    if (!keep.empty()) {
        qsa_histories.emplace(seq_id, std::move(keep));
    }
}

void llama_memory_hybrid_idx::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    llama_memory_hybrid::seq_add(seq_id, p0, p1, shift);

    if (mem_idx) {
        mem_idx->seq_add(seq_id, p0, p1, shift);
    }

    auto it = qsa_histories.find(seq_id);
    if (it != qsa_histories.end()) {
        for (auto & token : it->second) {
            if ((p0 < 0 || token.pos[0] >= p0) && (p1 < 0 || token.pos[0] < p1)) {
                token.pos[0] += shift;
            }
        }
    }
}

void llama_memory_hybrid_idx::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    llama_memory_hybrid::seq_div(seq_id, p0, p1, d);

    if (mem_idx) {
        mem_idx->seq_div(seq_id, p0, p1, d);
    }

    auto it = qsa_histories.find(seq_id);
    if (it != qsa_histories.end()) {
        for (auto & token : it->second) {
            if ((p0 < 0 || token.pos[0] >= p0) && (p1 < 0 || token.pos[0] < p1)) {
                token.pos[0] /= d;
            }
        }
    }
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_hybrid_idx::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> mb = llama_memory_hybrid::memory_breakdown();

    if (mem_idx) {
        for (const auto & buft_size : mem_idx->memory_breakdown()) {
            mb[buft_size.first] += buft_size.second;
        }
    }

    return mb;
}

void llama_memory_hybrid_idx::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    llama_memory_hybrid::state_write(io, seq_id, flags);

    // [TAG_HYBRID_IDX_STATE] the indexer section goes last, so it is a pure suffix: an old reader stops early instead of misparsing it
    // The indexer mirrors the attention cache, so it uses the same PARTIAL_ONLY gate.
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        if (mem_idx) {
            mem_idx->state_write(io, seq_id, flags);
        }

        uint32_t n_histories = 0;
        if (seq_id < 0) {
            n_histories = (uint32_t) qsa_histories.size();
        } else if (qsa_histories.count(seq_id) != 0) {
            n_histories = 1;
        }
        io.write(&n_histories, sizeof(n_histories));

        for (const auto & item : qsa_histories) {
            if (seq_id >= 0 && item.first != seq_id) {
                continue;
            }

            io.write(&item.first, sizeof(item.first));
            const uint64_t n_tokens = item.second.size();
            io.write(&n_tokens, sizeof(n_tokens));
            for (const auto & token : item.second) {
                io.write(token.pos.data(), sizeof(token.pos));
            }
        }
    }
}

void llama_memory_hybrid_idx::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    // note: repeats llama_memory_hybrid::state_read
    // the indexer needs the attention cache's cells, and a half-failed restore must leave all three caches alike

    // [TAG_HYBRID_IDX_SINFO]
    // the indexer restore adopts the attention cache's layout instead of searching for cells of its own
    // two find_slot calls agree only while both caches see the same occupancy, which a restore cannot promise
    llama_kv_cache::slot_info_vec_t sinfos_attn;

    try {
        if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
            get_mem_attn()->state_read_sinfo(io, seq_id, flags, mem_idx ? &sinfos_attn : nullptr, nullptr);
        }

        get_mem_recr()->state_read(io, seq_id, flags);

        // [TAG_HYBRID_IDX_STATE] must mirror the write order in state_write
        if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
            if (mem_idx) {
                mem_idx->state_read_sinfo(io, seq_id, flags, nullptr, &sinfos_attn);
            }

            uint32_t n_histories;
            io.read(&n_histories, sizeof(n_histories));
            if (n_histories > LLAMA_MAX_SEQ) {
                throw std::runtime_error("invalid QSA history count");
            }

            if (seq_id < 0) {
                qsa_histories.clear();
            } else {
                qsa_histories.erase(seq_id);
            }

            for (uint32_t ih = 0; ih < n_histories; ++ih) {
                llama_seq_id stored_seq;
                uint64_t n_tokens;
                io.read(&stored_seq, sizeof(stored_seq));
                io.read(&n_tokens, sizeof(n_tokens));
                if (stored_seq < 0 || stored_seq >= LLAMA_MAX_SEQ || n_tokens > get_mem_attn()->get_size()) {
                    throw std::runtime_error("invalid QSA history");
                }

                auto & history = qsa_histories[seq_id < 0 ? stored_seq : seq_id];
                history.resize(n_tokens);
                for (auto & token : history) {
                    io.read(token.pos.data(), sizeof(token.pos));
                }
            }
        }

    } catch (...) {
        // a half-restored context is the one state the indexer cannot fix by itself: attention holds new cells, the indexer old ones
        // drop what was being restored from all of them, which is a state they do agree on.
        state_drop(seq_id);

        throw;
    }
}

void llama_memory_hybrid_idx::state_drop(llama_seq_id seq_id) {
    // dropped directly, not via seq_rm: the recurrent cache may refuse it and then only the other two get cleared
    if (seq_id < 0) {
        clear(true);

        return;
    }

    get_mem_attn()->seq_rm(seq_id, -1, -1);
    get_mem_recr()->seq_rm(seq_id, -1, -1);

    if (mem_idx) {
        mem_idx->seq_rm(seq_id, -1, -1);
    }

    qsa_histories.erase(seq_id);
}

llama_kv_cache * llama_memory_hybrid_idx::get_mem_idx() const {
    return mem_idx.get();
}

void llama_memory_hybrid_idx::commit_qsa_tokens(const llama_ubatch & ubatch) {
    if (!mem_idx) {
        return;
    }

    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        qsa_token token = {};
        if (ubatch.token) {
            token.pos = { ubatch.pos[i], ubatch.pos[i], ubatch.pos[i], 0 };
        } else {
            for (uint32_t ip = 0; ip < token.pos.size(); ++ip) {
                token.pos[ip] = ip < ubatch.n_pos ? ubatch.pos[i + ip*ubatch.n_tokens] : ubatch.pos[i];
            }
        }

        for (int32_t is = 0; is < ubatch.n_seq_id[i]; ++is) {
            qsa_histories[ubatch.seq_id[i][is]].push_back(token);
        }
    }
}

void llama_memory_hybrid_idx::set_input_qsa(
        ggml_tensor *       block_cells,
        ggml_tensor *       block_pos,
        ggml_tensor *       block_mask,
        ggml_tensor *       tail_cells,
        ggml_tensor *       tail_fallback,
        const ggml_tensor * kq_mask,
        const llama_ubatch * ubatch,
        uint32_t            n_groups,
        uint32_t            ratio) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(block_cells->buffer));
    GGML_ASSERT(ggml_backend_buffer_is_host(block_pos->buffer));
    GGML_ASSERT(ggml_backend_buffer_is_host(block_mask->buffer));
    GGML_ASSERT(ggml_backend_buffer_is_host(tail_cells->buffer));
    GGML_ASSERT(ggml_backend_buffer_is_host(tail_fallback->buffer));
    GGML_ASSERT(ggml_backend_buffer_is_host(kq_mask->buffer));

    const int64_t n_blocks = block_cells->ne[1];
    const int64_t n_tokens = ubatch->n_tokens;
    const int64_t n_pos    = n_groups > 0 ? 4 : block_pos->ne[1];
    const int64_t n_kv     = kq_mask->ne[0];
    const int64_t n_tail   = tail_cells->ne[0];
    const int64_t n_layout = n_groups > 0 ? n_groups : n_tokens;

    GGML_ASSERT(block_cells->type == GGML_TYPE_I32);
    GGML_ASSERT(block_pos->type   == GGML_TYPE_I32);
    GGML_ASSERT(block_mask->type  == GGML_TYPE_F32);
    GGML_ASSERT(tail_cells->type  == GGML_TYPE_I32);
    GGML_ASSERT(tail_fallback->type == GGML_TYPE_F32);
    GGML_ASSERT(block_cells->ne[0] == ratio && block_cells->ne[2] == n_layout);
    if (n_groups > 0) {
        GGML_ASSERT(block_pos->ne[0] == n_blocks*n_pos && block_pos->ne[1] == n_layout);
    } else {
        GGML_ASSERT(block_pos->ne[0] == n_blocks && block_pos->ne[2] == n_layout);
    }
    GGML_ASSERT(block_mask->ne[0] == n_blocks && block_mask->ne[1] == n_tokens);
    GGML_ASSERT(n_tail == std::max<int64_t>(ratio - 1, 1) && tail_cells->ne[1] == n_tokens);
    GGML_ASSERT(tail_fallback->ne[0] == n_tail && tail_fallback->ne[1] == n_tokens);
    GGML_ASSERT(kq_mask->ne[0] == n_kv);
    GGML_ASSERT(n_groups == 0 || n_tokens % n_groups == 0);

    int32_t * cell_data = (int32_t *) block_cells->data;
    int32_t * pos_data  = (int32_t *) block_pos->data;
    float *   mask_data = (float *) block_mask->data;
    int32_t * tail_data = (int32_t *) tail_cells->data;
    float *   tail_fallback_data = (float *) tail_fallback->data;

    std::fill(cell_data, cell_data + ggml_nelements(block_cells), 0);
    std::fill(pos_data, pos_data + ggml_nelements(block_pos), 0);
    std::fill(mask_data, mask_data + ggml_nelements(block_mask), -INFINITY);
    std::fill(tail_data, tail_data + ggml_nelements(tail_cells), 0);
    std::fill(tail_fallback_data, tail_fallback_data + ggml_nelements(tail_fallback), 1.0f);

    auto mask_visible = [&](int64_t query, uint32_t cell) {
        const int64_t index = query*n_kv + cell;
        if (kq_mask->type == GGML_TYPE_F16) {
            return std::isfinite(ggml_fp16_to_fp32(((const ggml_fp16_t *) kq_mask->data)[index]));
        }
        return std::isfinite(((const float *) kq_mask->data)[index]);
    };

    using mapped_token = std::pair<const qsa_token *, uint32_t>;
    auto map_sequence = [&](llama_seq_id seq_id, std::vector<mapped_token> & mapped) {
        const auto found = qsa_histories.find(seq_id);
        if (found == qsa_histories.end()) {
            return false;
        }

        const auto & cells = get_mem_attn()->get_cells(seq_id);
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
        mapped.reserve(found->second.size());
        for (const auto & token : found->second) {
            const pos_key key = { token.pos[0], token.pos[1], token.pos[2] };
            auto cells = cells_by_pos.find(key);
            if (cells == cells_by_pos.end()) {
                continue;
            }

            size_t & index = next_cell[key];
            if (index >= cells->second.size()) {
                continue;
            }

            mapped.emplace_back(&token, cells->second[index++]);
        }

        return true;
    };

    if (n_groups > 0) {
        // Share one logical block table across the queries of each sequence.
        const int64_t n_qpg = n_tokens/n_groups;
        for (uint32_t ig = 0; ig < n_groups; ++ig) {
            const int64_t iq0 = ig*n_qpg;
            const llama_seq_id seq_id = ubatch->seq_id[iq0][0];
            std::vector<mapped_token> mapped;
            GGML_ASSERT(map_sequence(seq_id, mapped) && !mapped.empty());

            const size_t n_complete = mapped.size()/ratio;
            const size_t n_write = std::min<size_t>(n_complete, n_blocks);
            std::vector<uint8_t> used_cells(n_kv, 0);
            for (size_t ib = 0; ib < n_write; ++ib) {
                for (uint32_t ir = 0; ir < ratio; ++ir) {
                    const uint32_t cell = mapped[ib*ratio + ir].second;
                    cell_data[(ig*n_blocks + ib)*ratio + ir] = cell;
                    used_cells[cell] = 1;
                }
                for (int64_t ip = 0; ip < n_pos; ++ip) {
                    pos_data[(ig*n_pos + ip)*n_blocks + ib] = mapped[ib*ratio].first->pos[ip];
                }
            }

            uint32_t fallback_cell = 0;
            for (size_t ib = n_write; ib < (size_t) n_blocks; ++ib) {
                for (uint32_t ir = 0; ir < ratio; ++ir) {
                    while (fallback_cell < used_cells.size() && used_cells[fallback_cell]) {
                        ++fallback_cell;
                    }
                    GGML_ASSERT(fallback_cell < used_cells.size());
                    cell_data[(ig*n_blocks + ib)*ratio + ir] = fallback_cell;
                    used_cells[fallback_cell++] = 1;
                }
            }

            for (int64_t iq = iq0; iq < iq0 + n_qpg; ++iq) {
                const size_t n_visible = std::partition_point(mapped.begin(), mapped.end(), [&](const mapped_token & token) {
                    return mask_visible(iq, token.second);
                }) - mapped.begin();
                const size_t n_query_complete = n_visible/ratio;
                for (size_t ib = 0; ib < std::min<size_t>(n_query_complete, n_blocks); ++ib) {
                    mask_data[iq*n_blocks + ib] = 0.0f;
                }

                const size_t tail_start = n_query_complete*ratio;
                for (size_t iv = tail_start; iv < n_visible; ++iv) {
                    const size_t it = iq*n_tail + iv - tail_start;
                    tail_data[it] = mapped[iv].second;
                    tail_fallback_data[it] = 0.0f;
                }
            }
        }

        return;
    }

    for (int64_t iq = 0; iq < n_tokens; ++iq) {
        const llama_seq_id seq_id = ubatch->seq_id[iq][0];
        std::vector<mapped_token> mapped;
        if (!map_sequence(seq_id, mapped)) {
            continue;
        }

        std::vector<mapped_token> visible;
        visible.reserve(mapped.size());
        for (const auto & token : mapped) {
            if (mask_visible(iq, token.second)) {
                visible.push_back(token);
            }
        }

        const size_t n_complete = visible.size()/ratio;
        const size_t n_write = std::min<size_t>(n_complete, n_blocks);
        std::vector<uint8_t> used_cells(n_kv, 0);
        for (size_t ib = 0; ib < n_write; ++ib) {
            mask_data[iq*n_blocks + ib] = 0.0f;
            for (uint32_t ir = 0; ir < ratio; ++ir) {
                const uint32_t cell = visible[ib*ratio + ir].second;
                cell_data[(iq*n_blocks + ib)*ratio + ir] = cell;
                used_cells[cell] = 1;
            }
            for (int64_t ip = 0; ip < n_pos; ++ip) {
                pos_data[(iq*n_pos + ip)*n_blocks + ib] = visible[ib*ratio].first->pos[ip];
            }
        }

        uint32_t fallback_cell = 0;
        for (size_t ib = n_write; ib < (size_t) n_blocks; ++ib) {
            for (uint32_t ir = 0; ir < ratio; ++ir) {
                while (fallback_cell < used_cells.size() && used_cells[fallback_cell]) {
                    ++fallback_cell;
                }
                GGML_ASSERT(fallback_cell < used_cells.size());
                cell_data[(iq*n_blocks + ib)*ratio + ir] = fallback_cell;
                used_cells[fallback_cell++] = 1;
            }
        }

        const size_t tail_start = n_complete*ratio;
        for (size_t iv = tail_start; iv < visible.size(); ++iv) {
            const size_t it = iq*n_tail + iv - tail_start;
            tail_data[it] = visible[iv].second;
            tail_fallback_data[it] = 0.0f;
        }
    }
}

//
// llama_memory_hybrid_idx_context
//

// streams in each ubatch's slot info, matching get_k/get_v's `ns`
static std::vector<uint32_t> llama_memory_hybrid_idx_ns(const llama_kv_cache::slot_info_vec_t & sinfos) {
    std::vector<uint32_t> res;
    res.reserve(sinfos.size());

    for (const auto & sinfo : sinfos) {
        res.push_back(sinfo.s1 - sinfo.s0 + 1);
    }

    return res;
}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(llama_memory_status status) :
    llama_memory_hybrid_context(status) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(llama_memory_hybrid_idx * mem) :
    llama_memory_hybrid_context(mem),
    mem(mem),
    // graph reservation walks a full context, and qwen4exp builds the sparse attention only when this is set
    // without it the reserved worst case is the dense graph, so ggml-alloc must grow the buffer on the first decode
    ns_ubatch(mem->get_mem_idx() == nullptr ?
        std::vector<uint32_t>() : std::vector<uint32_t>{ mem->get_mem_idx()->get_n_stream() }),
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        new llama_kv_cache_context(mem->get_mem_idx())) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(
        llama_memory_hybrid_idx * mem,
                  llama_context * lctx,
                           bool   optimize) :
    llama_memory_hybrid_context(mem, lctx, optimize),
    mem(mem),
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        mem->get_mem_idx()->init_update(lctx, optimize)) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(
        llama_memory_hybrid_idx * mem,
                slot_info_vec_t   sinfos_attn,
                slot_info_vec_t   sinfos_idx,
      std::vector<llama_ubatch>   ubatches) :
    // note: the base copies the ubatches; ctx_idx gets a copy of its own
    llama_memory_hybrid_context(mem, std::move(sinfos_attn), ubatches),
    mem(mem),
    ns_ubatch(llama_memory_hybrid_idx_ns(sinfos_idx)),
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        new llama_kv_cache_context(mem->get_mem_idx(), std::move(sinfos_idx), ubatches)),
    has_ubatches(true) {}

bool llama_memory_hybrid_idx_context::next() {
    if (ctx_idx) {
        ctx_idx->next();
    }

    ++i_cur;

    return llama_memory_hybrid_context::next();
}

bool llama_memory_hybrid_idx_context::apply() {
    bool res = llama_memory_hybrid_context::apply();

    if (ctx_idx) {
        res = res & ctx_idx->apply();
    }

    if (res && ctx_idx && has_ubatches) {
        mem->commit_qsa_tokens(ctx_idx->get_ubatch());
    }

    return res;
}

const llama_kv_cache_context * llama_memory_hybrid_idx_context::get_idx() const {
    return static_cast<const llama_kv_cache_context *>(ctx_idx.get());
}

uint32_t llama_memory_hybrid_idx_context::get_n_stream() const {
    GGML_ASSERT(i_cur < ns_ubatch.size());

    return ns_ubatch[i_cur];
}

uint32_t llama_memory_hybrid_idx_context::get_qsa_compact_groups(const llama_ubatch & ubatch) const {
    // A shared table is exact when every query sees a prefix of one monotonic sequence history.
    if (!ubatch.equal_seqs() || ubatch.n_seqs == 0 || ubatch.n_seqs != ubatch.n_seqs_unq ||
            ubatch.n_tokens != ubatch.n_seq_tokens*ubatch.n_seqs) {
        return 0;
    }

    const uint32_t n_stream = get_n_stream();
    if (n_stream > 1 && ubatch.n_seqs != n_stream) {
        return 0;
    }

    std::vector<llama_seq_id> group_seqs;
    group_seqs.reserve(ubatch.n_seqs);
    std::map<llama_seq_id, bool> seen;
    for (uint32_t ig = 0; ig < ubatch.n_seqs; ++ig) {
        const uint32_t first = ig*ubatch.n_seq_tokens;
        if (ubatch.n_seq_id[first] != 1) {
            return 0;
        }

        const llama_seq_id seq_id = ubatch.seq_id[first][0];
        if (!seen.emplace(seq_id, true).second || (n_stream > 1 && seq_id != ubatch.seq_id_unq[ig])) {
            return 0;
        }

        for (uint32_t i = first + 1; i < first + ubatch.n_seq_tokens; ++i) {
            if (ubatch.n_seq_id[i] != 1 || ubatch.seq_id[i][0] != seq_id) {
                return 0;
            }
        }
        group_seqs.push_back(seq_id);
    }

    if (!has_ubatches) {
        return ubatch.n_seqs;
    }

    auto pos_less = [](const llama_memory_hybrid_idx::qsa_token & a,
                       const llama_memory_hybrid_idx::qsa_token & b) {
        return std::tie(a.pos[0], a.pos[1], a.pos[2]) < std::tie(b.pos[0], b.pos[1], b.pos[2]);
    };

    for (llama_seq_id seq_id : group_seqs) {
        const auto found = mem->qsa_histories.find(seq_id);
        if (found == mem->qsa_histories.end() || found->second.empty()) {
            return 0;
        }

        const auto & history = found->second;
        for (size_t i = 1; i < history.size(); ++i) {
            if (pos_less(history[i], history[i - 1])) {
                return 0;
            }
        }
    }

    return ubatch.n_seqs;
}

void llama_memory_hybrid_idx_context::set_input_qsa(
        ggml_tensor *       block_cells,
        ggml_tensor *       block_pos,
        ggml_tensor *       block_mask,
        ggml_tensor *       tail_cells,
        ggml_tensor *       tail_fallback,
        const ggml_tensor * kq_mask,
        const llama_ubatch * ubatch,
        uint32_t            n_groups,
        uint32_t            ratio) const {
    GGML_ASSERT(ratio > 0);
    GGML_ASSERT(mem != nullptr && mem->get_mem_idx() != nullptr);

    mem->set_input_qsa(block_cells, block_pos, block_mask, tail_cells, tail_fallback,
            kq_mask, ubatch, n_groups, ratio);
}
