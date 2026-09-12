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
    ggml_tensor * build_ids(ggml_context * ctx, ggml_backend_sched_t sched, ggml_tensor * ids, int layer) const;
    std::unique_lock<std::mutex> lock_graph();

    static bool handles(const ggml_tensor * node, void * data);
    static ggml_status compute(ggml_backend_t backend, ggml_tensor * node, void * data);

    void print_stats() const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
