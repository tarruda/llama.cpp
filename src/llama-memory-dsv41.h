#pragma once

#include "llama-memory.h"
#include "llama-hparams.h"
#include "ggml-cpp.h"

struct llama_model;

class llama_memory_dsv41 : public llama_memory_i {
public:
    struct layer {
        ggml_tensor * raw = nullptr;
        ggml_tensor * kv = nullptr;
        ggml_tensor * index = nullptr;
        ggml_tensor * comp_kv = nullptr;
        ggml_tensor * comp_score = nullptr;
    };

    llama_memory_dsv41(const llama_model & model, uint32_t n_ctx, uint32_t n_seq_max, uint32_t n_ubatch);

    llama_memory_context_ptr init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) override;
    llama_memory_context_ptr init_full() override;
    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;
    bool get_can_shift() const override { return false; }
    void clear(bool data) override;
    bool seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) override;
    void seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id) override;
    void seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) override;
    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;
    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;
    void state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const override;
    void state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) override;

    bool apply(const llama_ubatch & ubatch);
    void write_raw(int il, llama_pos pos, const float * values);
    void read_raw(int il, llama_pos pos, float * values) const;
    void write_kv(int il, llama_pos pos, const float * values);
    void read_kv(int il, llama_pos pos, float * values) const;
    void write_index(int il, llama_pos pos, const float * values);
    void read_index(int il, llama_pos pos, float * values) const;

    const llama_hparams hparams;
    const uint32_t n_ctx;
    const uint32_t n_ring;
    const uint32_t n_vocab;
    std::vector<layer> layers;
    std::vector<llama_token> tokens;

private:
    uint32_t ring_end = 0;
    ggml_context_ptr ctx;
    ggml_backend_buffer_ptr buffer;
};

class llama_memory_dsv41_context : public llama_memory_context_i {
public:
    llama_memory_dsv41_context(llama_memory_dsv41 * memory, std::vector<llama_ubatch> ubatches, llama_memory_status status);
    bool next() override;
    bool apply() override;
    const llama_ubatch & get_ubatch() const override;
    llama_memory_status get_status() const override { return status; }
    llama_memory_dsv41 * memory;

private:
    std::vector<llama_ubatch> ubatches;
    size_t index = 0;
    llama_memory_status status;
};
