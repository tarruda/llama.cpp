#include "llama-moe-stream.h"

#include "llama-model-loader.h"
#include "llama-hparams.h"
#include "llama-impl.h"
#include "llama.h"
#include "ggml-alloc.h"
#include "ggml-cpp.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <condition_variable>
#include <deque>
#include <fstream>
#include <future>
#include <limits>
#include <mutex>
#include <numeric>
#include <thread>
#include <unordered_map>

namespace {

class moe_read_pool {
public:
    explicit moe_read_pool(int n) {
        try {
            for (int i = 0; i < n; ++i) {
                workers.emplace_back([this] {
                    for (;;) {
                        std::packaged_task<void()> task;
                        {
                            std::unique_lock<std::mutex> lock(mutex);
                            ready.wait(lock, [&] { return stop || !tasks.empty(); });
                            if (tasks.empty()) { return; }
                            task = std::move(tasks.front());
                            tasks.pop_front();
                        }
                        task();
                    }
                });
            }
        } catch (...) {
            shutdown();
            throw;
        }
    }

    ~moe_read_pool() { shutdown(); }

    std::future<void> submit(std::function<void()> fn) {
        std::packaged_task<void()> task(std::move(fn));
        auto result = task.get_future();
        {
            std::lock_guard<std::mutex> lock(mutex);
            tasks.emplace_back(std::move(task));
        }
        ready.notify_one();
        return result;
    }

private:
    void shutdown() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stop = true;
        }
        ready.notify_all();
        for (auto & worker : workers) { worker.join(); }
        workers.clear();
    }

    std::mutex mutex;
    std::condition_variable ready;
    std::deque<std::packaged_task<void()>> tasks;
    std::vector<std::thread> workers;
    bool stop = false;
};

ggml_tensor * external_tensor(ggml_context * ctx, const ggml_tensor * tensor) {
    auto * result = ggml_dup_tensor(ctx, tensor);
    std::copy_n(tensor->nb, GGML_MAX_DIMS, result->nb);
    result->buffer = tensor->buffer;
    result->data = tensor->data;
    return result;
}

}

struct llama_moe_stream::impl {
    struct projection {
        ggml_tensor original = {};
        ggml_tensor * cache = nullptr;
        size_t offset = 0;
        uint32_t file = 0;
        bool direct_read = false;
    };

    struct decode_graph {
        ggml_tensor * prefetch = nullptr;
        ggml_tensor * ids = nullptr;
        ggml_tensor * gate = nullptr;
        ggml_tensor * up = nullptr;
        ggml_tensor * hidden = nullptr;
        ggml_tensor * down_input = nullptr;
        ggml_tensor * down = nullptr;
        int layer = -1;
        uint32_t resident_mask = 0;
        uint32_t missing_mask = 0;
        bool active = false;
    };

    struct layer {
        std::array<projection, 3> projections;
        std::vector<bool> protected_experts;
        std::vector<int32_t> expert_to_slot;
        std::vector<int32_t> slot_to_expert;
        std::vector<uint8_t> valid_projections;
        std::vector<uint64_t> age;
        std::array<std::vector<float>, 3> frequency;
        std::array<uint64_t, 3> observations = {};
        size_t expert_bytes = 0;
        int64_t n_slots = 0;
        bool fully_resident = false;
    };

    std::vector<layer> layers;
    ggml_context_ptr global_cache_ctx;
    ggml_backend_buffer_ptr global_cache_buffer;
    std::array<ggml_tensor *, 3> global_cache_tensors = {};
    ggml_context_ptr split_ctx;
    ggml_backend_buffer_ptr split_buffer;
    std::array<ggml_tensor *, 5> split_tensors = {};
    std::vector<int32_t> global_slot_to_layer;
    std::vector<int32_t> global_slot_to_expert;
    std::vector<uint8_t> global_valid_projections;
    std::vector<uint64_t> global_age;
    std::unordered_map<std::string, std::pair<int, int>> names;
    std::unordered_map<std::string, int> id_names;
    std::unordered_map<std::string, int> prefetch_names;
    std::unordered_map<const ggml_tensor *, decode_graph> decode_graphs;
    llama_files files;
    std::mutex mutex;
    std::mutex graph_mutex;
    std::mutex upload_mutex;
    std::unique_ptr<moe_read_pool> readers;
    int n_expert = 0;
    int n_expert_used = 0;
    int n_readers = 0;
    int64_t global_n_slots = 0;
    uint64_t tick = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t evictions = 0;
    uint64_t bytes_read = 0;
    uint64_t split_layers = 0;
    uint64_t split_resident_experts = 0;
    uint64_t split_missing_experts = 0;
    int64_t read_us = 0;
    int64_t read_span_us = 0;
    std::array<llama_moe_cache_stats, 3> phase_stats = {};
    bool check_tensors = false;
    bool global_cache = false;
    bool overlap = true;
    bool split_enabled = true;
    std::ofstream trace;
    llama_moe_phase phase = LLAMA_MOE_PHASE_PREFILL;
    llama_progress_callback progress = nullptr;
    void * progress_data = nullptr;
    using set_active_mask_t = void (*)(ggml_tensor *, uint32_t);
    set_active_mask_t set_active_mask = nullptr;

    impl(llama_model_loader & loader, const llama_hparams & hp, const llama_model_params & params) {
        if (loader.get_arch() != LLM_ARCH_DEEPSEEK41 || hp.n_layer() % 2 != 0) {
            throw std::runtime_error("routed expert streaming currently supports DeepSeek V4.1");
        }
        if (loader.files.empty()) {
            throw std::runtime_error("routed expert streaming requires a model loaded from GGUF files");
        }
#if defined(_WIN32)
        throw std::runtime_error("routed expert streaming currently requires POSIX positioned reads");
#endif
        if (params.split_mode == LLAMA_SPLIT_MODE_TENSOR || params.split_mode == LLAMA_SPLIT_MODE_ROW || params.load_mtp) {
            throw std::runtime_error("routed streaming does not support tensor/row splitting or MTP");
        }
        n_expert = hp.n_expert;
        if (n_expert <= 0) { throw std::runtime_error("routed streaming requires experts"); }
        for (uint32_t il = 0; il < hp.n_layer(); ++il) { n_expert_used = std::max(n_expert_used, (int) hp.n_expert_used(il)); }
        n_readers = 4;
        if (const char * value = std::getenv("LLAMA_MOE_OVERLAP")) { overlap = std::strcmp(value, "0") != 0; }
        if (const char * value = std::getenv("LLAMA_MOE_SPLIT")) { split_enabled = std::strcmp(value, "0") != 0; }
        split_enabled = split_enabled && overlap && n_expert_used <= 32;
        check_tensors = params.check_tensors;
        progress = params.progress_callback;
        progress_data = params.progress_callback_user_data;
        layers.resize(hp.n_layer());
        const std::array<llm_tensor, 3> kinds = {LLM_TENSOR_FFN_GATE_EXPS, LLM_TENSOR_FFN_UP_EXPS, LLM_TENSOR_FFN_DOWN_EXPS};
        const auto tn = LLM_TN(loader.get_arch());
        for (size_t il = 0; il < layers.size(); ++il) {
            auto & l = layers[il];
            id_names.emplace("moe_stream_ids." + std::to_string(il), il);
            prefetch_names.emplace("moe_stream_prefetch." + std::to_string(il), il);
            l.protected_experts.assign(n_expert, false);
            for (size_t p = 0; p < kinds.size(); ++p) {
                const auto name = tn(kinds[p], "weight", il).str();
                auto * t = loader.require_tensor_meta(name);
                if (!ggml_is_contiguous(t) || t->ne[2] != n_expert || t->ne[3] != 1) {
                    throw std::runtime_error("unsupported routed tensor layout: " + name);
                }
                auto & source = l.projections[p];
                source.original = *t;
                source.original.buffer = nullptr;
                source.original.data = nullptr;
                if (!loader.files.empty()) {
                    const auto & w = loader.require_weight(name.c_str());
                    source.file = w.idx;
                    source.offset = w.offs;
                } else if (!loader.no_alloc) {
                    throw std::runtime_error("routed streaming requires GGUF source files");
                }
                l.expert_bytes += t->nb[2];
                names.emplace(name, std::make_pair(il, p));
            }
        }

        uint64_t budget = params.moe_cache_bytes;
        const uint64_t expert_bytes = layers.front().expert_bytes;
        if (!budget) { budget = std::max(1, n_expert_used)*expert_bytes; }
        const uint64_t total_routed_bytes = std::accumulate(layers.begin(), layers.end(), uint64_t(0), [=](uint64_t sum, const layer & l) { return sum + n_expert*l.expert_bytes; });
        const uint64_t total_experts = layers.size()*n_expert;
        global_cache = budget < total_routed_bytes;
        if (global_cache && std::any_of(layers.begin(), layers.end(), [=](const layer & l) { return l.expert_bytes != expert_bytes; })) {
            throw std::runtime_error("adaptive MoE caching requires equal expert sizes in every layer");
        }
        const uint64_t available_slots = std::min(total_experts, budget/expert_bytes);
        const int resident_layers = global_cache && layers.size() > 19 && available_slots >= 12*(uint64_t) n_expert ? 4 : 0;
        if (resident_layers) {
            static constexpr int order[] = { 0, 1, 19, 2 };
            for (int i = 0; i < resident_layers; ++i) { layers[order[i]].fully_resident = true; }
        }
        const uint64_t resident_bytes = (uint64_t) resident_layers*n_expert*expert_bytes;
        uint64_t minimum = 0;
        for (auto & l : layers) {
            l.n_slots = l.fully_resident ? n_expert : 1;
            minimum += l.n_slots*l.expert_bytes;
        }
        if (global_cache) {
            const uint64_t resident_experts = resident_bytes/expert_bytes;
            const uint64_t streamed_experts = total_experts - resident_experts;
            minimum = (resident_experts + (streamed_experts > 0))*expert_bytes;
        }
        if (budget < minimum) {
            throw std::runtime_error(format("MoE cache requires at least %.2f MiB for its working slots; budget is %.2f MiB", minimum/1048576.0, budget/1048576.0));
        }
        uint64_t allocated = 0;
        if (global_cache) {
            const uint64_t resident_experts = resident_bytes/expert_bytes;
            global_n_slots = std::min(total_experts - resident_experts, budget/expert_bytes - resident_experts);
            allocated = (global_n_slots + resident_experts)*expert_bytes;
            global_slot_to_layer.assign(global_n_slots, -1);
            global_slot_to_expert.assign(global_n_slots, -1);
            global_valid_projections.assign(global_n_slots, 0);
            global_age.assign(global_n_slots, 0);
        } else {
            allocated = minimum;
            bool added = true;
            while (added) {
                added = false;
                for (auto & l : layers) {
                    if (l.n_slots < n_expert && l.expert_bytes <= budget - allocated) {
                        ++l.n_slots;
                        allocated += l.expert_bytes;
                        added = true;
                    }
                }
            }
        }
        for (auto & l : layers) {
            if (!global_cache) { l.fully_resident = l.n_slots == n_expert; }
            l.expert_to_slot.assign(n_expert, -1);
            if (!uses_global(l)) {
                l.slot_to_expert.assign(l.n_slots, -1);
                l.valid_projections.assign(l.n_slots, 0);
                l.age.assign(l.n_slots, 0);
            }
            for (auto & counts : l.frequency) { counts.assign(n_expert, 0); }
        }
        LLAMA_LOG_INFO("moe_stream: adaptive cache%s\n", global_cache ? " with a global slot pool" : "");
        LLAMA_LOG_INFO("moe_stream: %d resident layers, %.2f MiB resident, %.2f MiB dynamic, %.2f MiB total routed cache, %d read threads\n",
                resident_layers, resident_bytes/1048576.0, (allocated - resident_bytes)/1048576.0, allocated/1048576.0, n_readers);
    }

    bool uses_global(const layer & l) const {
        return global_cache && !l.fully_resident;
    }

    struct expert_load {
        int32_t expert;
        int32_t slot;
        uint8_t projections;
    };

    struct read_batch {
        layer * target = nullptr;
        std::vector<expert_load> loads;
        std::vector<std::future<void>> tasks;
        llama_moe_phase phase;
        uint64_t bytes = 0;
        int64_t start = 0;
        bool prefetch = false;
    } pending;

    void observe(int il, const std::vector<int32_t> & original, int used) {
        auto & l = layers[il];
        if (trace.is_open()) {
            trace << nlohmann::json({{"layer", il}, {"phase", phase}, {"used", used}, {"ids", original}}).dump() << '\n';
            if (!trace) { throw std::runtime_error("cannot write MoE route trace"); }
        }
        auto & counts = l.frequency[phase];
        for (size_t i = 0; i < original.size(); ++i) {
            counts[original[i]] += 1.0f;
            if ((i + 1) % used == 0 && ++l.observations[phase] % 16 == 0) {
                for (auto & count : counts) { count *= 0.5f; }
            }
        }
    }

    size_t count_protected() const {
        size_t count = 0;
        for (const auto & l : layers) { if (!l.fully_resident) { count += std::count(l.protected_experts.begin(), l.protected_experts.end(), true); } }
        return count;
    }

    uint64_t expert_age(const layer & l, int expert) const {
        const int slot = l.expert_to_slot[expert];
        return slot < 0 ? 0 : uses_global(l) ? global_age[slot] : l.age[slot];
    }

    void select_phase_protection() {
        for (auto & l : layers) { std::fill(l.protected_experts.begin(), l.protected_experts.end(), false); }
        if (!global_cache || phase == LLAMA_MOE_PHASE_DECODE) { return; }
        struct ranked_expert { int layer; int expert; };
        std::vector<ranked_expert> hot;
        for (size_t il = 0; il < layers.size(); ++il) {
            const auto & l = layers[il];
            if (l.fully_resident) { continue; }
            for (int e = 0; e < n_expert; ++e) {
                if (!l.protected_experts[e] && l.expert_to_slot[e] >= 0 && l.frequency[LLAMA_MOE_PHASE_DECODE][e] > 0) { hot.push_back({(int) il, e}); }
            }
        }
        std::sort(hot.begin(), hot.end(), [&](const ranked_expert & a, const ranked_expert & b) {
            const auto & la = layers[a.layer];
            const auto & lb = layers[b.layer];
            const float ca = la.frequency[LLAMA_MOE_PHASE_DECODE][a.expert];
            const float cb = lb.frequency[LLAMA_MOE_PHASE_DECODE][b.expert];
            if (ca != cb) { return ca > cb; }
            const uint64_t aa = expert_age(la, a.expert);
            const uint64_t ab = expert_age(lb, b.expert);
            return aa != ab ? aa > ab : std::tie(a.layer, a.expert) < std::tie(b.layer, b.expert);
        });
        const size_t dynamic = global_n_slots - count_protected();
        const size_t reserve = std::min(dynamic, std::max<size_t>(n_expert, (dynamic + 9)/10));
        const size_t keep = std::min(hot.size(), dynamic - reserve);
        for (size_t i = 0; i < keep; ++i) { layers[hot[i].layer].protected_experts[hot[i].expert] = true; }
    }

    void fit_small_group(layer & l, const std::vector<int32_t> & experts) {
        if (!global_cache || phase == LLAMA_MOE_PHASE_DECODE) { return; }
        const int available = global_n_slots;
        if ((int) experts.size() > available) { return; }
        std::vector<bool> required(n_expert, false);
        int needed = 0;
        for (int e : experts) { required[e] = true; needed += !l.protected_experts[e]; }
        int capacity = global_n_slots - count_protected();
        if (capacity >= needed) { return; }
        struct ranked_expert { int layer; int expert; };
        std::vector<ranked_expert> candidates;
        for (size_t il = 0; il < layers.size(); ++il) {
            auto & candidate_layer = layers[il];
            if (candidate_layer.fully_resident) { continue; }
            for (int e = 0; e < n_expert; ++e) {
                if (candidate_layer.protected_experts[e] && (&candidate_layer != &l || !required[e])) {
                    candidates.push_back({(int) il, e});
                }
            }
        }
        std::sort(candidates.begin(), candidates.end(), [&](const ranked_expert & a, const ranked_expert & b) {
            const auto & la = layers[a.layer];
            const auto & lb = layers[b.layer];
            const float ca = la.frequency[LLAMA_MOE_PHASE_DECODE][a.expert];
            const float cb = lb.frequency[LLAMA_MOE_PHASE_DECODE][b.expert];
            if (ca != cb) { return ca < cb; }
            return expert_age(la, a.expert) < expert_age(lb, b.expert);
        });
        for (const auto & candidate : candidates) {
            if (capacity >= needed) { break; }
            layers[candidate.layer].protected_experts[candidate.expert] = false;
            ++capacity;
        }
    }

    bool expert_resident(const layer & l, int expert) const {
        const int slot = l.expert_to_slot[expert];
        if (slot < 0) { return false; }
        return (uses_global(l) ? global_valid_projections[slot] : l.valid_projections[slot]) == 7;
    }

    void set_decode_mask(decode_graph & decode, uint32_t mask) const {
        set_active_mask(decode.gate, mask);
        set_active_mask(decode.up, mask);
        set_active_mask(decode.hidden, mask);
        set_active_mask(decode.down, mask);
    }

    static void bind_split_tensor(ggml_tensor & target, const ggml_tensor * source, ggml_tensor * storage) {
        GGML_ASSERT(ggml_nbytes(source) <= ggml_nbytes(storage));
        target = *source;
        target.buffer = storage->buffer;
        target.data = storage->data;
        target.view_src = nullptr;
        target.view_offs = 0;
    }

    ggml_status launch_resident(ggml_backend_t backend, decode_graph & decode) {
        ggml_tensor gate, up, hidden, quant, down;
        bind_split_tensor(gate, decode.gate, split_tensors[0]);
        bind_split_tensor(up, decode.up, split_tensors[1]);
        bind_split_tensor(hidden, decode.hidden, split_tensors[2]);
        bind_split_tensor(down, decode.down, split_tensors[4]);
        hidden.src[0] = &gate;
        hidden.src[1] = &up;
        ggml_tensor * activation = &hidden;
        const bool quantized = decode.down_input != decode.hidden;
        if (quantized) {
            bind_split_tensor(quant, decode.down_input, split_tensors[3]);
            quant.src[0] = &hidden;
            activation = &quant;
        }
        down.src[1] = activation;
        set_active_mask(&gate, decode.resident_mask);
        set_active_mask(&up, decode.resident_mask);
        set_active_mask(&hidden, decode.resident_mask);
        set_active_mask(&down, decode.resident_mask);

        ggml_context_ptr ctx(ggml_init({ggml_graph_overhead_custom(8, false), nullptr, true}));
        if (!ctx) { return GGML_STATUS_ALLOC_FAILED; }
        auto * graph = ggml_new_graph_custom(ctx.get(), 8, false);
        ggml_graph_add_node(graph, &gate);
        ggml_graph_add_node(graph, &up);
        ggml_graph_add_node(graph, &hidden);
        if (quantized) { ggml_graph_add_node(graph, &quant); }
        ggml_graph_add_node(graph, &down);
        return ggml_backend_graph_compute_async(backend, graph);
    }

    ggml_status copy_resident_output(ggml_backend_t backend, const decode_graph & decode) {
        ggml_tensor resident;
        bind_split_tensor(resident, decode.down, split_tensors[4]);
        ggml_backend_tensor_copy_async(backend, backend, &resident, decode.down);
        return GGML_STATUS_SUCCESS;
    }

    void begin_fill(layer & l, const std::vector<expert_load> & loads, bool prefetch) {
        GGML_ASSERT(!pending.target);
        if (loads.empty()) { return; }
        pending.loads = loads;
        pending.tasks.reserve(loads.size()*l.projections.size());
        pending.target = &l;
        pending.phase = phase;
        pending.start = ggml_time_us();
        pending.prefetch = prefetch;
        try {
            for (const auto & load : loads) {
                for (size_t index = 0; index < l.projections.size(); ++index) {
                    if (!(load.projections & (1 << index))) { continue; }
                    auto * p = &l.projections[index];
                    pending.bytes += p->original.nb[2];
                    pending.tasks.push_back(readers->submit([this, p, load] {
                        const size_t size = p->original.nb[2];
                        if (p->direct_read) {
                            auto * target = (uint8_t *) p->cache->data + load.slot*size;
                            files.at(p->file)->read_at(target, size, p->offset + load.expert*size);
                            if (check_tensors && !ggml_validate_row_data(p->original.type, target, size)) {
                                throw std::runtime_error("invalid routed expert data");
                            }
                            return;
                        }
                        std::vector<uint8_t> staging(size);
                        files.at(p->file)->read_at(staging.data(), size, p->offset + load.expert*size);
                        if (check_tensors && !ggml_validate_row_data(p->original.type, staging.data(), size)) {
                            throw std::runtime_error("invalid routed expert data");
                        }
                        std::lock_guard<std::mutex> lock(upload_mutex);
                        ggml_backend_tensor_set(p->cache, staging.data(), load.slot*size, size);
                    }));
                }
            }
        } catch (...) {
            finish_fill(std::current_exception());
        }
    }

    void finish_fill(std::exception_ptr failure = nullptr) {
        if (!pending.target) { return; }
        auto batch = std::move(pending);
        pending = {};
        const int64_t wait_start = ggml_time_us();
        for (auto & result : batch.tasks) {
            try { result.get(); } catch (...) { if (!failure) { failure = std::current_exception(); } }
        }
        const auto now = ggml_time_us();
        const auto wait = now - wait_start;
        read_us += wait;
        read_span_us += now - batch.start;
        phase_stats[batch.phase].read_ms += wait/1000.0;
        if (failure) { std::rethrow_exception(failure); }
        bytes_read += batch.bytes;
        phase_stats[batch.phase].bytes_read += batch.bytes;
        auto & l = *batch.target;
        if (trace.is_open()) {
            trace << nlohmann::json({{"event", "read"}, {"layer", &l - layers.data()}, {"phase", batch.phase},
                    {"bytes", batch.bytes}, {"span_us", now - batch.start}, {"wait_us", wait}, {"prefetch", batch.prefetch}}).dump() << '\n';
        }
        for (const auto & load : batch.loads) {
            if (uses_global(l)) {
                global_valid_projections[load.slot] |= load.projections;
                global_age[load.slot] = ++tick;
            } else {
                l.valid_projections[load.slot] |= load.projections;
                l.age[load.slot] = ++tick;
            }
        }
    }

    void acquire(layer & l, const std::vector<int32_t> & experts, uint8_t projections = 7, bool prefetch = false) {
        const int il = &l - layers.data();
        const bool shared = uses_global(l);
        const int64_t n_slots = shared ? global_n_slots : l.n_slots;
        std::vector<bool> protect(n_expert, false);
        for (int32_t e : experts) { protect[e] = true; }
        std::vector<expert_load> loads;
        std::vector<bool> reserved(n_slots, false);
        for (int32_t e : experts) {
            const int current = l.expert_to_slot[e];
            const uint8_t wanted = l.protected_experts[e] ? 7 : projections;
            if (current >= 0) {
                if (shared) { global_age[current] = ++tick; }
                else { l.age[current] = ++tick; }
                const uint8_t valid = shared ? global_valid_projections[current] : l.valid_projections[current];
                const uint8_t missing = wanted & ~valid;
                if (missing) {
                    loads.push_back({e, current, missing});
                    ++misses;
                    ++phase_stats[phase].misses;
                } else {
                    ++hits;
                    ++phase_stats[phase].hits;
                }
                continue;
            }
            int victim = -1;
            for (int slot = 0; slot < n_slots; ++slot) {
                const int old_layer = shared ? global_slot_to_layer[slot] : il;
                const int old = shared ? global_slot_to_expert[slot] : l.slot_to_expert[slot];
                if (reserved[slot] || (old >= 0 && (layers[old_layer].protected_experts[old] || (old_layer == il && protect[old])))) { continue; }
                const int previous_layer = victim < 0 ? -1 : shared ? global_slot_to_layer[victim] : il;
                const int previous = victim < 0 ? -1 : shared ? global_slot_to_expert[victim] : l.slot_to_expert[victim];
                if (victim < 0 || old < 0 || (previous >= 0 && (
                        layers[old_layer].frequency[phase][old] != layers[previous_layer].frequency[phase][previous] ?
                        layers[old_layer].frequency[phase][old] < layers[previous_layer].frequency[phase][previous] :
                        (shared ? global_age[slot] < global_age[victim] : l.age[slot] < l.age[victim])))) {
                    victim = slot;
                }
                if (old < 0) { break; }
            }
            if (victim < 0) { throw std::runtime_error("MoE expert group exceeds available cache slots"); }
            const int old_layer = shared ? global_slot_to_layer[victim] : il;
            const int old = shared ? global_slot_to_expert[victim] : l.slot_to_expert[victim];
            if (old >= 0) { layers[old_layer].expert_to_slot[old] = -1; ++evictions; ++phase_stats[phase].evictions; }
            if (shared) {
                global_slot_to_layer[victim] = -1;
                global_slot_to_expert[victim] = -1;
                global_valid_projections[victim] = 0;
            } else {
                l.slot_to_expert[victim] = -1;
                l.valid_projections[victim] = 0;
            }
            reserved[victim] = true;
            ++misses;
            ++phase_stats[phase].misses;
            loads.push_back({e, victim, wanted});
        }
        for (const auto & load : loads) {
            l.expert_to_slot[load.expert] = load.slot;
            if (shared) {
                global_slot_to_layer[load.slot] = il;
                global_slot_to_expert[load.slot] = load.expert;
            } else {
                l.slot_to_expert[load.slot] = load.expert;
            }
        }
        begin_fill(l, loads, prefetch);
        if (!prefetch) { finish_fill(); }
    }

    ggml_status execute(ggml_backend_t backend, ggml_tensor * node) {
        std::lock_guard<std::mutex> lock(mutex);
        const auto split_down = node->op == GGML_OP_MUL_MAT_ID && node->src[2] && node->src[2]->src[0] ? decode_graphs.find(node->src[2]->src[0]) : decode_graphs.end();
        if (split_down != decode_graphs.end() && split_down->second.down == node) {
            auto & decode = split_down->second;
            GGML_ASSERT(decode.active);
            auto status = copy_resident_output(backend, decode);
            if (status == GGML_STATUS_SUCCESS) {
                ggml_context_ptr ctx(ggml_init({ggml_graph_overhead_custom(2, false), nullptr, true}));
                if (!ctx) { status = GGML_STATUS_ALLOC_FAILED; }
                else {
                    ggml_tensor operation = *node;
                    auto * graph = ggml_new_graph_custom(ctx.get(), 2, false);
                    ggml_graph_add_node(graph, &operation);
                    status = ggml_backend_graph_compute_async(backend, graph);
                }
            }
            decode.active = false;
            return status;
        }
        if (node->op == GGML_OP_VIEW) {
            auto & l = layers[id_names.at(node->name)];
            GGML_ASSERT(!pending.target || pending.target == &l);
            finish_fill();
            const auto graph = node->src[0] ? decode_graphs.find(node->src[0]) : decode_graphs.end();
            if (graph != decode_graphs.end() && graph->second.active) {
                auto & decode = graph->second;
                set_decode_mask(decode, decode.missing_mask);
                ++split_layers;
                split_resident_experts += __builtin_popcount(decode.resident_mask);
                split_missing_experts += __builtin_popcount(decode.missing_mask);
            }
            return GGML_STATUS_SUCCESS;
        }
        if (node->op == GGML_OP_DUP) {
            const bool prefetch = prefetch_names.count(node->name);
            const int il = prefetch ? prefetch_names.at(node->name) : id_names.at(node->name);
            auto & l = layers[il];
            decode_graph * decode = nullptr;
            const auto graph = decode_graphs.find(node);
            if (graph != decode_graphs.end()) { decode = &graph->second; }
            ggml_backend_synchronize(backend);
            if (decode && decode->gate) {
                set_decode_mask(*decode, 0);
                decode->resident_mask = 0;
                decode->missing_mask = 0;
                decode->active = false;
            }
            const auto * ids = node->src[0];
            std::vector<uint8_t> raw(ggml_nbytes(ids));
            ggml_backend_tensor_get(ids, raw.data(), 0, raw.size());
            std::vector<int32_t> original;
            for (int64_t token = 0; token < ids->ne[1]; ++token) {
                for (int64_t rank = 0; rank < ids->ne[0]; ++rank) {
                    int32_t e;
                    memcpy(&e, raw.data() + token*ids->nb[1] + rank*ids->nb[0], sizeof(e));
                    if (e < 0 || e >= n_expert) { throw std::runtime_error("invalid original routed expert ID"); }
                    original.push_back(e);
                }
            }
            observe(il, original, ids->ne[0]);
            auto required = original;
            std::sort(required.begin(), required.end());
            required.erase(std::unique(required.begin(), required.end()), required.end());
            fit_small_group(l, required);
            if (prefetch && phase == LLAMA_MOE_PHASE_DECODE && decode && decode->gate && original.size() <= 32) {
                const uint32_t all = original.size() == 32 ? UINT32_MAX : (1u << original.size()) - 1u;
                for (size_t i = 0; i < original.size(); ++i) {
                    if (expert_resident(l, original[i])) { decode->resident_mask |= 1u << i; }
                }
                decode->missing_mask = all & ~decode->resident_mask;
                decode->active = decode->resident_mask != 0 && decode->missing_mask != 0;
            }
            acquire(l, required, 7, prefetch);
            for (auto & e : original) { e = l.expert_to_slot[e]; }
            ggml_backend_tensor_set(node, original.data(), 0, original.size()*sizeof(int32_t));
            if (decode && decode->active) {
                const auto status = launch_resident(backend, *decode);
                if (status != GGML_STATUS_SUCCESS) { return status; }
            }
            return GGML_STATUS_SUCCESS;
        }
        const auto location = names.at(node->src[0]->name);
        auto & l = layers[location.first];
        auto * cache = l.projections[location.second].cache;
        if (node->src[0] != cache || !ggml_backend_supports_buft(backend, ggml_backend_buffer_get_type(cache->buffer))) {
            throw std::runtime_error("streamed expert matmul moved to an incompatible backend");
        }
        if (node->src[1]->type != GGML_TYPE_F32 || !ggml_is_contiguous(node->src[1]) || !ggml_is_contiguous(node)) {
            throw std::runtime_error("unsupported streamed expert activation layout");
        }
        ggml_backend_synchronize(backend);
        const auto * ids = node->src[2];
        std::vector<uint8_t> raw(ggml_nbytes(ids));
        ggml_backend_tensor_get(ids, raw.data(), 0, raw.size());
        const int64_t used = ids->ne[0], tokens = ids->ne[1];
        std::vector<int32_t> original(used*tokens);
        std::vector<std::vector<int32_t>> rows(n_expert);
        for (int64_t token = 0; token < tokens; ++token) {
            for (int64_t rank = 0; rank < used; ++rank) {
                int32_t e;
                memcpy(&e, raw.data() + token*ids->nb[1] + rank*ids->nb[0], sizeof(e));
                if (e < 0 || e >= n_expert) { throw std::runtime_error("invalid original routed expert ID"); }
                const int32_t row = token*used + rank;
                original[row] = e;
                rows[e].push_back(row);
            }
        }
        std::vector<int32_t> required;
        if (location.second == 0) { observe(location.first, original, used); }
        for (int e = 0; e < n_expert; ++e) { if (!rows[e].empty()) { required.push_back(e); } }
        fit_small_group(l, required);
        const int protected_count = global_cache ? count_protected() : std::count(l.protected_experts.begin(), l.protected_experts.end(), true);
        const int capacity = (global_cache ? global_n_slots : l.n_slots) - protected_count;
        std::vector<std::vector<int32_t>> groups(1);
        int unpinned = 0;
        for (int e : required) {
            if (!l.protected_experts[e] && unpinned == capacity) { groups.emplace_back(); unpinned = 0; }
            groups.back().push_back(e);
            unpinned += !l.protected_experts[e];
        }
        for (const auto & group : groups) {
            acquire(l, group, groups.size() == 1 ? 7 : 1 << location.second);
            ggml_context_ptr ctx(ggml_init({256*ggml_tensor_overhead() + ggml_graph_overhead_custom(64, false), nullptr, true}));
            if (!ctx) { return GGML_STATUS_ALLOC_FAILED; }
            if (groups.size() == 1) {
                auto * mapped = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, used, tokens);
                std::vector<int32_t> slots(original.size());
                for (size_t i = 0; i < slots.size(); ++i) { slots[i] = l.expert_to_slot[original[i]]; }
                ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
                if (!buffer) { return GGML_STATUS_ALLOC_FAILED; }
                ggml_backend_tensor_set(mapped, slots.data(), 0, slots.size()*sizeof(int32_t));
                ggml_tensor operation = *node;
                operation.src[2] = mapped;
                auto * graph = ggml_new_graph_custom(ctx.get(), 64, false);
                ggml_graph_add_node(graph, &operation);
                const auto status = ggml_backend_graph_compute(backend, graph);
                if (status != GGML_STATUS_SUCCESS) { return status; }
            } else {
                std::vector<int32_t> input_rows, output_rows, slot_ids;
                for (int e : group) {
                    for (int32_t row : rows[e]) {
                        output_rows.push_back(row);
                        input_rows.push_back((row/used)*node->src[1]->ne[1] + (row%used)%node->src[1]->ne[1]);
                        slot_ids.push_back(l.expert_to_slot[e]);
                    }
                }
                const int64_t count = output_rows.size();
                auto * input_index = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, count);
                auto * output_index = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, count);
                auto * mapped = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 1, count);
                auto * input = external_tensor(ctx.get(), node->src[1]);
                input = ggml_reshape_2d(ctx.get(), input, input->ne[0], input->ne[1]*input->ne[2]);
                auto * packed = ggml_get_rows(ctx.get(), input, input_index);
                packed = ggml_reshape_3d(ctx.get(), packed, packed->ne[0], 1, count);
                auto * product = ggml_mul_mat_id(ctx.get(), cache, packed, mapped);
                memcpy(product->op_params, node->op_params, sizeof(product->op_params));
                using preserve_dispatch_t = void (*)(ggml_tensor *, const ggml_tensor *);
                auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
                auto preserve_dispatch = (preserve_dispatch_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_metal_mul_mat_id_preserve_dispatch");
                if (preserve_dispatch) { preserve_dispatch(product, node); }
                product = ggml_reshape_2d(ctx.get(), product, product->ne[0], count);
                auto * output = external_tensor(ctx.get(), node);
                output = ggml_reshape_2d(ctx.get(), output, output->ne[0], used*tokens);
                auto * scatter = ggml_set_rows(ctx.get(), output, product, output_index);
                auto * graph = ggml_new_graph_custom(ctx.get(), 64, false);
                ggml_build_forward_expand(graph, scatter);
                ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
                if (!buffer) { return GGML_STATUS_ALLOC_FAILED; }
                ggml_backend_tensor_set(input_index, input_rows.data(), 0, count*sizeof(int32_t));
                ggml_backend_tensor_set(output_index, output_rows.data(), 0, count*sizeof(int32_t));
                ggml_backend_tensor_set(mapped, slot_ids.data(), 0, count*sizeof(int32_t));
                const auto status = ggml_backend_graph_compute(backend, graph);
                if (status != GGML_STATUS_SUCCESS) { return status; }
            }
        }
        return GGML_STATUS_SUCCESS;
    }
};

llama_moe_stream::llama_moe_stream(llama_model_loader & loader, const llama_hparams & hparams, const llama_model_params & params) :
    pimpl(std::make_unique<impl>(loader, hparams, params)) {}

llama_moe_stream::~llama_moe_stream() { synchronize(); print_stats(); }

int64_t llama_moe_stream::slots(const char * name) const {
    const auto it = pimpl->names.find(name);
    if (it == pimpl->names.end()) { return 0; }
    const auto & l = pimpl->layers[it->second.first];
    return pimpl->uses_global(l) ? 1 : l.n_slots;
}

void llama_moe_stream::bind(ggml_tensor * tensor) {
    const auto it = pimpl->names.find(tensor->name);
    if (it != pimpl->names.end()) { pimpl->layers[it->second.first].projections[it->second.second].cache = tensor; }
}

const ggml_tensor * llama_moe_stream::source(const ggml_tensor * tensor) const {
    const auto it = pimpl->names.find(tensor->name);
    return it == pimpl->names.end() ? tensor : &pimpl->layers[it->second.first].projections[it->second.second].original;
}

bool llama_moe_stream::load(llama_files & files) {
    pimpl->files = std::move(files);
    pimpl->readers = std::make_unique<moe_read_pool>(pimpl->n_readers);
    if (pimpl->global_cache) {
        const ggml_init_params params = { 4*ggml_tensor_overhead(), nullptr, true };
        pimpl->global_cache_ctx.reset(ggml_init(params));
        if (!pimpl->global_cache_ctx) { throw std::runtime_error("failed to create global MoE cache context"); }
        const auto dynamic = std::find_if(pimpl->layers.begin(), pimpl->layers.end(), [&](const auto & l) { return pimpl->uses_global(l); });
        GGML_ASSERT(dynamic != pimpl->layers.end());
        auto * buft = ggml_backend_buffer_get_type(dynamic->projections.front().cache->buffer);
        for (size_t p = 0; p < pimpl->global_cache_tensors.size(); ++p) {
            const auto * base = dynamic->projections[p].cache;
            auto * tensor = ggml_dup_tensor(pimpl->global_cache_ctx.get(), base);
            tensor->ne[2] = pimpl->global_n_slots;
            tensor->nb[3] = tensor->ne[2]*tensor->nb[2];
            ggml_format_name(tensor, "moe_stream_global.%zu", p);
            pimpl->global_cache_tensors[p] = tensor;
            for (const auto & l : pimpl->layers) {
                const auto * cache = l.projections[p].cache;
                if (cache->type != tensor->type || cache->ne[0] != tensor->ne[0] || cache->ne[1] != tensor->ne[1] || cache->nb[2] != tensor->nb[2] ||
                        ggml_backend_buffer_get_type(cache->buffer) != buft) {
                    throw std::runtime_error("global adaptive MoE cache requires matching routed tensors and buffer types");
                }
            }
        }
        pimpl->global_cache_buffer.reset(ggml_backend_alloc_ctx_tensors_from_buft(pimpl->global_cache_ctx.get(), buft));
        if (!pimpl->global_cache_buffer) { throw std::runtime_error("failed to allocate global MoE cache"); }
        ggml_backend_buffer_set_usage(pimpl->global_cache_buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        for (auto & l : pimpl->layers) {
            if (!pimpl->uses_global(l)) { continue; }
            for (size_t p = 0; p < l.projections.size(); ++p) {
                auto * cache = l.projections[p].cache;
                const auto * global = pimpl->global_cache_tensors[p];
                std::copy_n(global->ne, GGML_MAX_DIMS, cache->ne);
                std::copy_n(global->nb, GGML_MAX_DIMS, cache->nb);
                cache->buffer = global->buffer;
                cache->data = global->data;
            }
        }
        LLAMA_LOG_INFO("moe_stream: %s global cache buffer size = %.2f MiB\n", ggml_backend_buffer_name(pimpl->global_cache_buffer.get()),
                ggml_backend_buffer_get_size(pimpl->global_cache_buffer.get())/1048576.0);
    }
    if (const char * path = std::getenv("LLAMA_MOE_TRACE")) {
        pimpl->trace.open(path);
        if (!pimpl->trace) { throw std::runtime_error("cannot open MoE route trace"); }
        nlohmann::json layout = nlohmann::json::array();
        for (const auto & l : pimpl->layers) {
            layout.push_back({{"slots", l.n_slots}, {"expert_bytes", l.expert_bytes}, {"fully_resident", l.fully_resident}});
        }
        pimpl->trace << nlohmann::json({{"format", "llama-moe-trace"}, {"version", 1}, {"experts", pimpl->n_expert},
                {"global_slots", pimpl->global_n_slots}, {"layers", layout}}).dump() << '\n';
    }
    for (auto & l : pimpl->layers) {
        for (auto & p : l.projections) {
            auto * buffer = p.cache->buffer;
            p.direct_read = ggml_backend_buffer_is_host(buffer);
            auto * dev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(buffer));
            if (dev && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU && !p.direct_read) {
                throw std::runtime_error("routed streaming requires native CPU buffers; repacked expert buffers are not supported");
            }
            if (dev && !p.direct_read) {
                auto * reg = ggml_backend_dev_backend_reg(dev);
                using get_host_ptr_t = void * (*)(ggml_backend_buffer_t);
                auto get_host_ptr = (get_host_ptr_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_metal_buffer_get_host_ptr");
                p.direct_read = get_host_ptr && get_host_ptr(buffer);
            }
        }
    }
    if (pimpl->split_enabled) {
        auto * buft = ggml_backend_buffer_get_type(pimpl->layers.front().projections.front().cache->buffer);
        auto * dev = ggml_backend_buft_get_device(buft);
        auto * reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
        pimpl->set_active_mask = reg ? (impl::set_active_mask_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_metal_set_moe_active_mask") : nullptr;
        pimpl->split_enabled = pimpl->set_active_mask && std::all_of(pimpl->layers.begin(), pimpl->layers.end(), [=](const auto & l) {
            return std::all_of(l.projections.begin(), l.projections.end(), [=](const auto & p) {
                return p.direct_read && ggml_backend_buffer_get_type(p.cache->buffer) == buft;
            });
        });
        if (pimpl->split_enabled) {
            int64_t hidden = 0, output = 0;
            for (const auto & l : pimpl->layers) {
                hidden = std::max({hidden, l.projections[0].original.ne[1], l.projections[1].original.ne[1]});
                output = std::max(output, l.projections[2].original.ne[1]);
            }
            pimpl->split_ctx.reset(ggml_init({6*ggml_tensor_overhead(), nullptr, true}));
            if (!pimpl->split_ctx) { throw std::runtime_error("failed to create split MoE context"); }
            for (int i = 0; i < 4; ++i) {
                pimpl->split_tensors[i] = ggml_new_tensor_3d(pimpl->split_ctx.get(), GGML_TYPE_F32, hidden, pimpl->n_expert_used, 1);
                ggml_format_name(pimpl->split_tensors[i], "moe_stream_split.%d", i);
            }
            pimpl->split_tensors[4] = ggml_new_tensor_3d(pimpl->split_ctx.get(), GGML_TYPE_F32, output, pimpl->n_expert_used, 1);
            ggml_set_name(pimpl->split_tensors[4], "moe_stream_split.output");
            pimpl->split_buffer.reset(ggml_backend_alloc_ctx_tensors_from_buft(pimpl->split_ctx.get(), buft));
            if (!pimpl->split_buffer) { throw std::runtime_error("failed to allocate split MoE workspace"); }
            ggml_backend_buffer_set_usage(pimpl->split_buffer.get(), GGML_BACKEND_BUFFER_USAGE_COMPUTE);
            LLAMA_LOG_INFO("moe_stream: resident/missing overlap workspace = %.2f MiB\n", ggml_backend_buffer_get_size(pimpl->split_buffer.get())/1048576.0);
        }
    }
    for (auto & l : pimpl->layers) {
        for (int e = 0; e < pimpl->n_expert; ++e) {
            if (!l.protected_experts[e] && !l.fully_resident) { continue; }
            if (pimpl->progress && !pimpl->progress(1.0f, pimpl->progress_data)) { return false; }
            pimpl->acquire(l, {e});
        }
    }
    return true;
}

bool llama_moe_stream::handles(const ggml_tensor * node, void * data) {
    const auto & self = *static_cast<llama_moe_stream *>(data);
    if ((node->op == GGML_OP_DUP || node->op == GGML_OP_VIEW) && self.pimpl->id_names.count(node->name)) { return true; }
    if (node->op == GGML_OP_DUP && self.pimpl->prefetch_names.count(node->name)) { return true; }
    if (node->op != GGML_OP_MUL_MAT_ID || !node->src[0]) { return false; }
    const auto split_down = node->src[2] && node->src[2]->src[0] ? self.pimpl->decode_graphs.find(node->src[2]->src[0]) : self.pimpl->decode_graphs.end();
    if (split_down != self.pimpl->decode_graphs.end() && split_down->second.down == node && split_down->second.active) { return true; }
    if (self.pimpl->id_names.count(node->src[2]->name)) { return false; }
    const auto it = self.pimpl->names.find(node->src[0]->name);
    return it != self.pimpl->names.end() && !self.pimpl->layers[it->second.first].fully_resident;
}

void llama_moe_stream::set_phase(llama_moe_phase phase) {
    if (phase < LLAMA_MOE_PHASE_CALIBRATION || phase > LLAMA_MOE_PHASE_DECODE) {
        throw std::invalid_argument("invalid MoE execution phase");
    }
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    pimpl->finish_fill();
    if (pimpl->phase == phase) { return; }
    pimpl->phase = phase;
    pimpl->select_phase_protection();
}

ggml_status llama_moe_stream::compute(ggml_backend_t backend, ggml_tensor * node, void * data) {
    try {
        return static_cast<llama_moe_stream *>(data)->pimpl->execute(backend, node);
    } catch (const std::exception & error) {
        LLAMA_LOG_ERROR("moe_stream: %s: %s\n", node->name, error.what());
        return GGML_STATUS_FAILED;
    }
}

void llama_moe_stream::print_stats() const {
    if (!pimpl->readers) { return; }
    LLAMA_LOG_INFO("moe_stream: %llu hits, %llu misses, %llu evictions, %.3f GiB read, %.3f s waiting (%.3f s load intervals)\n",
            (unsigned long long) pimpl->hits, (unsigned long long) pimpl->misses, (unsigned long long) pimpl->evictions,
            pimpl->bytes_read/1073741824.0, pimpl->read_us/1e6, pimpl->read_span_us/1e6);
    if (pimpl->split_layers) {
        LLAMA_LOG_INFO("moe_stream: %llu resident/missing splits, %llu resident experts, %llu missing experts\n",
                (unsigned long long) pimpl->split_layers, (unsigned long long) pimpl->split_resident_experts,
                (unsigned long long) pimpl->split_missing_experts);
    }
}

llama_moe_cache_stats llama_moe_stream::stats(llama_moe_phase phase) const {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    return pimpl->phase_stats.at(phase);
}

std::unique_lock<std::mutex> llama_moe_stream::lock_graph() {
    return std::unique_lock<std::mutex>(pimpl->graph_mutex);
}

bool llama_moe_stream::synchronize() {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    try {
        pimpl->finish_fill();
        return true;
    } catch (const std::exception & error) {
        LLAMA_LOG_ERROR("moe_stream: %s\n", error.what());
        return false;
    }
}

void llama_moe_stream::begin_graph() {
    pimpl->decode_graphs.clear();
}

ggml_tensor * llama_moe_stream::build_ids(ggml_context * ctx, ggml_backend_sched_t sched, ggml_cgraph * graph, ggml_tensor * ids, ggml_tensor * shared, ggml_tensor * expert_input, ggml_tensor * weights, int il) {
    auto & l = pimpl->layers.at(il);
    if (l.fully_resident) { return ids; }
    const int capacity = pimpl->global_cache ? pimpl->global_n_slots : l.n_slots;
    if (ggml_nelements(ids) > capacity) { return ids; }
    auto * buft = ggml_backend_buffer_get_type(l.projections[0].cache->buffer);
    auto * dev = ggml_backend_buft_get_device(buft);
    if (!dev || !ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(dev), "ggml_backend_metal_buffer_get_host_ptr")) { return ids; }
    for (const auto & p : l.projections) {
        if (ggml_backend_buffer_get_type(p.cache->buffer) != buft) { return ids; }
    }
    for (int i = 0; i < ggml_backend_sched_get_n_backends(sched); ++i) {
        auto * backend = ggml_backend_sched_get_backend(sched, i);
        if (ggml_backend_get_device(backend) != dev) { continue; }
        const bool prefetch = pimpl->overlap && std::all_of(l.projections.begin(), l.projections.end(), [](const auto & p) { return p.direct_read; });
        auto * mapped = ggml_dup(ctx, ids);
        ggml_set_name(mapped, ((prefetch ? "moe_stream_prefetch." : "moe_stream_ids.") + std::to_string(il)).c_str());
        ggml_backend_sched_set_tensor_backend(sched, mapped, backend);
        if (prefetch) {
            const bool split = pimpl->split_enabled && ids->ne[1] == 1 && expert_input && weights;
            if (split) {
                mapped->src[1] = expert_input;
                mapped->src[2] = weights;
            }
            ggml_build_forward_expand(graph, mapped);
            ggml_build_forward_expand(graph, shared);
            // The readiness marker aliases the IDs; it must not overwrite live GPU scratch.
            auto * ready = ggml_view_2d(ctx, mapped, mapped->ne[0], mapped->ne[1], mapped->nb[1], 0);
            ggml_set_name(ready, ("moe_stream_ids." + std::to_string(il)).c_str());
            ggml_backend_sched_set_tensor_backend(sched, ready, backend);
            if (split) {
                auto & decode = pimpl->decode_graphs[mapped];
                decode = {};
                decode.prefetch = mapped;
                decode.ids = ready;
                decode.layer = il;
            }
            return ready;
        }
        return mapped;
    }
    return ids;
}

void llama_moe_stream::register_decode_graph(int il, ggml_tensor * ids, ggml_tensor * gate, ggml_tensor * up, ggml_tensor * hidden, ggml_tensor * down_input, ggml_tensor * down) {
    const auto graph = ids->src[0] ? pimpl->decode_graphs.find(ids->src[0]) : pimpl->decode_graphs.end();
    if (graph == pimpl->decode_graphs.end()) { return; }
    auto & decode = graph->second;
    const bool valid = il == decode.layer && ids == decode.ids && gate->op == GGML_OP_MUL_MAT_ID && up->op == GGML_OP_MUL_MAT_ID && down->op == GGML_OP_MUL_MAT_ID &&
            gate->src[2] == ids && up->src[2] == ids && down->src[2] == ids && hidden->op == GGML_OP_DSV41_SWIGLU &&
            hidden->src[0] == gate && hidden->src[1] == up && down->src[1] == down_input &&
            (down_input == hidden || (down_input->op == GGML_OP_DSV41_ACT_QUANT && down_input->src[0] == hidden)) &&
            ggml_nbytes(gate) <= ggml_nbytes(pimpl->split_tensors[0]) && ggml_nbytes(up) <= ggml_nbytes(pimpl->split_tensors[1]) &&
            ggml_nbytes(hidden) <= ggml_nbytes(pimpl->split_tensors[2]) && ggml_nbytes(down_input) <= ggml_nbytes(pimpl->split_tensors[3]) &&
            ggml_nbytes(down) <= ggml_nbytes(pimpl->split_tensors[4]);
    if (!valid) {
        pimpl->decode_graphs.erase(graph);
        return;
    }
    decode.gate = gate;
    decode.up = up;
    decode.hidden = hidden;
    decode.down_input = down_input;
    decode.down = down;
    pimpl->set_decode_mask(decode, 0);
}
