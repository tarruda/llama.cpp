#pragma once

#include "llama-batch.h"
#include "llama-kv-cache-msa.h"
#include "llama-memory.h"
#include "llama-memory-recurrent.h"

#include <array>
#include <map>
#include <memory>
#include <vector>

class llama_memory_qwen4 : public llama_memory_i {
public:
    llama_memory_qwen4(
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
                     bool   mtp);

    ~llama_memory_qwen4() = default;

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    bool get_can_shift() const override;

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0)       override;

    llama_kv_cache_msa * get_mem_attn() const;
    llama_kv_cache * get_mem_qsa() const;
    llama_memory_recurrent * get_mem_gdn() const;
    llama_memory_recurrent * get_mem_ple() const;

private:
    friend class llama_memory_qwen4_context;

    struct token_entry {
        std::array<llama_pos, 4> pos;
        llama_token token;
    };

    using token_history = std::vector<token_entry>;

    bool prepare_ple_ids(const llama_ubatch & ubatch, std::vector<int32_t> & ids) const;
    void commit_tokens(const llama_ubatch & ubatch);
    void set_input_qsa_layout(
            ggml_tensor * block_cells,
            ggml_tensor * block_key_cells,
            ggml_tensor * block_mask,
            ggml_tensor * selected,
            ggml_tensor * tail_cells,
            ggml_tensor * update_cells,
            ggml_tensor * update_pos,
            ggml_tensor * update_idxs,
            const ggml_tensor * kq_mask,
            const llama_ubatch * ubatch,
            uint32_t ratio,
            uint32_t block_topk,
            uint32_t n_kv,
            uint32_t n_stream) const;

    uint32_t get_qsa_update_capacity(
            const llama_ubatch & ubatch,
            uint32_t ratio,
            uint32_t n_blocks,
            bool reserve) const;

    const llama_hparams & hparams;
    llama_hparams hparams_qsa;
    const std::unique_ptr<llama_kv_cache_msa> mem_attn;
    const std::unique_ptr<llama_kv_cache> mem_qsa;
    const std::unique_ptr<llama_memory_recurrent> mem_gdn;
    const std::unique_ptr<llama_memory_recurrent> mem_ple;

    std::map<llama_seq_id, token_history> token_histories;
    mutable std::map<llama_seq_id, uint32_t> qsa_cached_blocks;
};

class llama_memory_qwen4_context : public llama_memory_context_i {
public:
    using slot_info_vec_t = llama_kv_cache::slot_info_vec_t;

    explicit llama_memory_qwen4_context(llama_memory_status status);
    explicit llama_memory_qwen4_context(llama_memory_qwen4 * mem);
    llama_memory_qwen4_context(llama_memory_qwen4 * mem, llama_context * lctx, bool optimize);
    llama_memory_qwen4_context(
            llama_memory_qwen4 * mem,
            slot_info_vec_t sinfos_base,
            slot_info_vec_t sinfos_idx,
            std::vector<llama_ubatch> ubatches);

    ~llama_memory_qwen4_context() = default;

    bool next() override;
    bool apply() override;

    llama_memory_status get_status() const override;
    const llama_ubatch & get_ubatch() const override;

    const llama_kv_cache_msa_context * get_attn() const;
    llama_kv_cache_msa * get_attn_memory() const;
    llama_kv_cache * get_qsa() const;
    const llama_memory_recurrent_context * get_gdn() const;
    const llama_memory_recurrent_context * get_ple() const;

    void set_input_ple_ids(ggml_tensor * dst) const;
    void set_input_qsa_layout(
            ggml_tensor * block_cells,
            ggml_tensor * block_key_cells,
            ggml_tensor * block_mask,
            ggml_tensor * selected,
            ggml_tensor * tail_cells,
            ggml_tensor * update_cells,
            ggml_tensor * update_pos,
            ggml_tensor * update_idxs,
            const ggml_tensor * kq_mask,
            const llama_ubatch * ubatch,
            uint32_t ratio,
            uint32_t block_topk,
            uint32_t n_kv,
            uint32_t n_stream) const;

    uint32_t get_qsa_update_capacity(
            const llama_ubatch & ubatch,
            uint32_t ratio,
            uint32_t n_blocks) const;

private:
    llama_memory_qwen4 * mem = nullptr;

    size_t i_next = 0;
    std::vector<llama_ubatch> ubatches;
    std::vector<int32_t> ple_ids;

    const llama_memory_context_ptr ctx_attn;
    const llama_memory_context_ptr ctx_gdn;
    const llama_memory_context_ptr ctx_ple;

    const bool reserve = false;
    const llama_memory_status status;
};
