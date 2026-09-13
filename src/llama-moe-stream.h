#pragma once

#include "ggml-backend.h"
#include "llama-mmap.h"
#include "llama.h"

#include <memory>
#include <mutex>

struct llama_model_loader;
struct llama_hparams;
struct llama_model_params;

class llama_moe_stream {
public:
    llama_moe_stream(llama_model_loader & loader, const llama_hparams & hparams, const llama_model_params & params);
    ~llama_moe_stream();

    int64_t slots(const char * name) const;
    void bind(ggml_tensor * tensor);
    const ggml_tensor * source(const ggml_tensor * tensor) const;
    bool load(llama_files & files);
    void set_phase(llama_moe_phase phase);
    llama_moe_cache_stats stats(llama_moe_phase phase) const;
    ggml_tensor * build_ids(ggml_context * ctx, ggml_backend_sched_t sched, ggml_cgraph * graph, ggml_tensor * ids, ggml_tensor * shared, ggml_tensor * expert_input, ggml_tensor * weights, int layer);
    void begin_graph();
    void register_decode_graph(int layer, ggml_tensor * ids, ggml_tensor * gate, ggml_tensor * up, ggml_tensor * hidden, ggml_tensor * down_input, ggml_tensor * down);
    std::unique_lock<std::mutex> lock_graph();
    bool synchronize();

    static bool handles(const ggml_tensor * node, void * data);
    static ggml_status compute(ggml_backend_t backend, ggml_tensor * node, void * data);

    void print_stats() const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
