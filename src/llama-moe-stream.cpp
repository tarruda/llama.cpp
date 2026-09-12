#include "llama-moe-stream.h"

#include "llama-model-loader.h"
#include "llama-hparams.h"
#include "llama-impl.h"
#include "llama.h"
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

uint64_t profile_uint(const nlohmann::json & value) {
    if (!value.is_number_unsigned() && !(value.is_number_integer() && value.get<int64_t>() >= 0)) {
        throw std::runtime_error("MoE profile integers must be nonnegative");
    }
    return value.get<uint64_t>();
}

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

    struct layer {
        std::array<projection, 3> projections;
        std::vector<bool> pinned;
        std::array<std::vector<bool>, 3> phase_pins;
        std::vector<int32_t> expert_to_slot;
        std::vector<int32_t> slot_to_expert;
        std::vector<uint8_t> valid_projections;
        std::vector<uint64_t> age;
        size_t expert_bytes = 0;
        int64_t n_slots = 0;
        bool fully_resident = false;
    };

    std::vector<layer> layers;
    std::unordered_map<std::string, std::pair<int, int>> names;
    std::unordered_map<std::string, int> id_names;
    llama_files files;
    std::mutex mutex;
    std::mutex graph_mutex;
    std::mutex upload_mutex;
    std::unique_ptr<moe_read_pool> readers;
    int n_expert = 0;
    int n_readers = 0;
    uint64_t tick = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t evictions = 0;
    uint64_t bytes_read = 0;
    int64_t read_us = 0;
    std::array<llama_moe_cache_stats, 3> phase_stats = {};
    bool check_tensors = false;
    llama_moe_phase phase = LLAMA_MOE_PHASE_PREFILL;
    llama_progress_callback progress = nullptr;
    void * progress_data = nullptr;

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
        if (params.moe_pin_count < 0 || params.moe_read_threads < 1 || params.moe_read_threads > 64) {
            throw std::runtime_error("invalid MoE pin count or read thread count");
        }
        n_expert = hp.n_expert;
        if (n_expert <= 0 || (uint64_t) params.moe_pin_count > (uint64_t) hp.n_layer()*n_expert) {
            throw std::runtime_error("MoE pin count exceeds the model expert count");
        }
        n_readers = params.moe_read_threads;
        check_tensors = params.check_tensors;
        progress = params.progress_callback;
        progress_data = params.progress_callback_user_data;
        layers.resize(hp.n_layer());
        const std::array<llm_tensor, 3> kinds = {LLM_TENSOR_FFN_GATE_EXPS, LLM_TENSOR_FFN_UP_EXPS, LLM_TENSOR_FFN_DOWN_EXPS};
        const auto tn = LLM_TN(loader.get_arch());
        for (size_t il = 0; il < layers.size(); ++il) {
            auto & l = layers[il];
            id_names.emplace("moe_stream_ids." + std::to_string(il), il);
            l.pinned.assign(n_expert, params.moe_pin_encoder && il < layers.size()/2);
            l.phase_pins.fill(l.pinned);
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

        const int encoder_pins = params.moe_pin_encoder ? int(layers.size()/2)*n_expert : 0;
        const int ranked_pins = std::max(0, params.moe_pin_count - encoder_pins);
        if (params.moe_pin_count > 0 && (!params.moe_profile || !params.moe_profile[0])) {
            throw std::runtime_error("a positive MoE pin count requires a routing profile");
        }
        if (params.moe_profile && params.moe_profile[0]) {
            std::ifstream input(params.moe_profile);
            if (!input) { throw std::runtime_error("cannot open MoE routing profile"); }
            const auto doc = nlohmann::json::parse(input);
            const auto & model = doc.at("model");
            if (doc.at("format") != "llama-moe-profile" || profile_uint(doc.at("version")) != 1 ||
                    model.at("architecture") != loader.get_arch_name() || profile_uint(model.at("layers")) != layers.size() ||
                    profile_uint(model.at("experts_per_layer")) != (uint64_t) n_expert || profile_uint(model.at("embedding_length")) != hp.n_embd) {
                throw std::runtime_error("MoE profile does not match the model architecture and expert topology");
            }
            const std::array<const char *, 3> phases = {"calibration", "prefill", "decode"};
            for (size_t phase_id = 0; phase_id < phases.size(); ++phase_id) {
                struct ranked_expert { uint64_t count; int layer; int expert; };
                std::vector<ranked_expert> ranked;
                std::vector<bool> seen(layers.size()*n_expert, false);
                const bool has_phase = doc.contains("phases") && doc.at("phases").contains(phases[phase_id]);
                if (!has_phase && ranked_pins > 0) {
                    LLAMA_LOG_WARN("moe_stream: no %s observations; using calibration counts for these pins\n", phases[phase_id]);
                }
                const auto & observations = has_phase ?
                    doc.at("phases").at(phases[phase_id]).at("layers") : doc.at("layers");
                for (const auto & row : observations) {
                    const uint64_t il = profile_uint(row.at("layer"));
                    if (il >= layers.size()) { throw std::runtime_error("MoE profile layer is out of range"); }
                    for (const auto & expert : row.at("experts")) {
                        const uint64_t id = profile_uint(expert.at("id"));
                        if (id >= (uint64_t) n_expert || seen[il*n_expert + id]) {
                            throw std::runtime_error("MoE profile has duplicate or invalid experts");
                        }
                        seen[il*n_expert + id] = true;
                        const uint64_t count = profile_uint(expert.at("count"));
                        if (!layers[il].pinned[id]) { ranked.push_back({count, (int) il, (int) id}); }
                    }
                }
                if (std::find(seen.begin(), seen.end(), false) != seen.end()) {
                    throw std::runtime_error("MoE profile must include counts for every expert, including zero counts");
                }
                std::sort(ranked.begin(), ranked.end(), [](const ranked_expert & a, const ranked_expert & b) {
                    if (a.count != b.count) { return a.count > b.count; }
                    return std::tie(a.layer, a.expert) < std::tie(b.layer, b.expert);
                });
                for (int i = 0; i < ranked_pins; ++i) { layers[ranked[i].layer].phase_pins[phase_id][ranked[i].expert] = true; }
            }
        }

        for (auto & l : layers) { l.pinned = l.phase_pins[phase]; }

        uint64_t budget = params.moe_cache_bytes;
        if (!budget) {
            size_t free = 0, total = 0;
            auto * cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
            ggml_backend_dev_memory(cpu, &free, &total);
            budget = (free ? free : total)/3*2;
        }
        uint64_t minimum = 0, pinned_bytes = 0;
        size_t total_pins = 0;
        for (auto & l : layers) {
            const int pins = std::count(l.pinned.begin(), l.pinned.end(), true);
            total_pins += pins;
            pinned_bytes += pins*l.expert_bytes;
            int max_pins = pins;
            for (const auto & mask : l.phase_pins) { max_pins = std::max(max_pins, (int) std::count(mask.begin(), mask.end(), true)); }
            l.n_slots = std::min(n_expert, max_pins + 1);
            minimum += l.n_slots*l.expert_bytes;
        }
        if (budget < minimum) {
            throw std::runtime_error(format("MoE cache requires at least %.2f MiB for requested pins and dynamic slots; budget is %.2f MiB", minimum/1048576.0, budget/1048576.0));
        }
        uint64_t allocated = minimum;
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
        for (auto & l : layers) {
            l.fully_resident = l.n_slots == n_expert;
            l.expert_to_slot.assign(n_expert, -1);
            l.slot_to_expert.assign(l.n_slots, -1);
            l.valid_projections.assign(l.n_slots, 0);
            l.age.assign(l.n_slots, 0);
        }
        LLAMA_LOG_INFO("moe_stream: %zu pins (%d encoder preset), %.2f MiB pinned, %.2f MiB dynamic, %.2f MiB total routed cache, %d read threads\n",
                total_pins, encoder_pins, pinned_bytes/1048576.0, (allocated - pinned_bytes)/1048576.0, allocated/1048576.0, n_readers);
    }

    struct expert_load {
        int32_t expert;
        int32_t slot;
        uint8_t projections;
    };

    void fill(layer & l, const std::vector<expert_load> & loads) {
        if (loads.empty()) { return; }
        std::vector<std::future<void>> pending;
        pending.reserve(loads.size()*l.projections.size());
        const int64_t start = ggml_time_us();
        uint64_t bytes = 0;
        std::exception_ptr failure;
        try {
            for (const auto & load : loads) {
                for (size_t index = 0; index < l.projections.size(); ++index) {
                    if (!(load.projections & (1 << index))) { continue; }
                    auto * p = &l.projections[index];
                    bytes += p->original.nb[2];
                    pending.push_back(readers->submit([this, p, load] {
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
            failure = std::current_exception();
        }
        for (auto & result : pending) {
            try { result.get(); } catch (...) { if (!failure) { failure = std::current_exception(); } }
        }
        const auto elapsed = ggml_time_us() - start;
        read_us += elapsed;
        phase_stats[phase].read_ms += elapsed/1000.0;
        if (failure) { std::rethrow_exception(failure); }
        bytes_read += bytes;
        phase_stats[phase].bytes_read += bytes;
        for (const auto & load : loads) {
            l.expert_to_slot[load.expert] = load.slot;
            l.slot_to_expert[load.slot] = load.expert;
            l.valid_projections[load.slot] |= load.projections;
            l.age[load.slot] = ++tick;
        }
    }

    void acquire(layer & l, const std::vector<int32_t> & experts, uint8_t projections = 7) {
        std::vector<bool> protect(n_expert, false);
        for (int32_t e : experts) { protect[e] = true; }
        std::vector<expert_load> loads;
        std::vector<bool> reserved(l.n_slots, false);
        for (int32_t e : experts) {
            const int current = l.expert_to_slot[e];
            const uint8_t wanted = l.pinned[e] ? 7 : projections;
            if (current >= 0) {
                l.age[current] = ++tick;
                const uint8_t missing = wanted & ~l.valid_projections[current];
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
            for (int slot = 0; slot < l.n_slots; ++slot) {
                const int old = l.slot_to_expert[slot];
                if (reserved[slot] || (old >= 0 && (l.pinned[old] || protect[old]))) { continue; }
                if (victim < 0 || l.age[slot] < l.age[victim]) { victim = slot; }
                if (old < 0) { break; }
            }
            if (victim < 0) { throw std::runtime_error("MoE expert group exceeds available cache slots"); }
            const int old = l.slot_to_expert[victim];
            if (old >= 0) { l.expert_to_slot[old] = -1; ++evictions; ++phase_stats[phase].evictions; }
            l.slot_to_expert[victim] = -1;
            l.valid_projections[victim] = 0;
            reserved[victim] = true;
            ++misses;
            ++phase_stats[phase].misses;
            loads.push_back({e, victim, wanted});
        }
        fill(l, loads);
    }

    ggml_status execute(ggml_backend_t backend, ggml_tensor * node) {
        std::lock_guard<std::mutex> lock(mutex);
        if (node->op == GGML_OP_DUP) {
            auto & l = layers[id_names.at(node->name)];
            ggml_backend_synchronize(backend);
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
            auto required = original;
            std::sort(required.begin(), required.end());
            required.erase(std::unique(required.begin(), required.end()), required.end());
            acquire(l, required);
            for (auto & e : original) { e = l.expert_to_slot[e]; }
            ggml_backend_tensor_set(node, original.data(), 0, original.size()*sizeof(int32_t));
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
        for (int e = 0; e < n_expert; ++e) { if (!rows[e].empty()) { required.push_back(e); } }
        const int pins = std::count(l.pinned.begin(), l.pinned.end(), true);
        const int capacity = l.n_slots - pins;
        std::vector<std::vector<int32_t>> groups(1);
        int unpinned = 0;
        for (int e : required) {
            if (!l.pinned[e] && unpinned == capacity) { groups.emplace_back(); unpinned = 0; }
            groups.back().push_back(e);
            unpinned += !l.pinned[e];
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

llama_moe_stream::~llama_moe_stream() { print_stats(); }

int64_t llama_moe_stream::slots(const char * name) const {
    const auto it = pimpl->names.find(name);
    return it == pimpl->names.end() ? 0 : pimpl->layers[it->second.first].n_slots;
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
        for (int e = 0; e < pimpl->n_expert; ++e) {
            if (!l.pinned[e] && !l.fully_resident) { continue; }
            if (pimpl->progress && !pimpl->progress(1.0f, pimpl->progress_data)) { return false; }
            pimpl->acquire(l, {e});
        }
    }
    return true;
}

bool llama_moe_stream::handles(const ggml_tensor * node, void * data) {
    const auto & self = *static_cast<llama_moe_stream *>(data);
    if (node->op == GGML_OP_DUP && self.pimpl->id_names.count(node->name)) { return true; }
    if (node->op != GGML_OP_MUL_MAT_ID || !node->src[0]) { return false; }
    if (self.pimpl->id_names.count(node->src[2]->name)) { return false; }
    const auto it = self.pimpl->names.find(node->src[0]->name);
    return it != self.pimpl->names.end() && !self.pimpl->layers[it->second.first].fully_resident;
}

void llama_moe_stream::set_phase(llama_moe_phase phase) {
    if (phase < LLAMA_MOE_PHASE_CALIBRATION || phase > LLAMA_MOE_PHASE_DECODE) {
        throw std::invalid_argument("invalid MoE execution phase");
    }
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    if (pimpl->phase == phase) { return; }
    for (auto & l : pimpl->layers) { l.pinned = l.phase_pins[phase]; }
    pimpl->phase = phase;
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
    LLAMA_LOG_INFO("moe_stream: %llu hits, %llu misses, %llu evictions, %.3f GiB read in %.3f s\n",
            (unsigned long long) pimpl->hits, (unsigned long long) pimpl->misses, (unsigned long long) pimpl->evictions,
            pimpl->bytes_read/1073741824.0, pimpl->read_us/1e6);
}

llama_moe_cache_stats llama_moe_stream::stats(llama_moe_phase phase) const {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    return pimpl->phase_stats.at(phase);
}

std::unique_lock<std::mutex> llama_moe_stream::lock_graph() {
    return std::unique_lock<std::mutex>(pimpl->graph_mutex);
}

ggml_tensor * llama_moe_stream::build_ids(ggml_context * ctx, ggml_backend_sched_t sched, ggml_tensor * ids, int il) const {
    const auto & l = pimpl->layers.at(il);
    if (l.fully_resident) { return ids; }
    int capacity = l.n_slots;
    for (const auto & pins : l.phase_pins) {
        capacity = std::min(capacity, int(l.n_slots) - int(std::count(pins.begin(), pins.end(), true)));
    }
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
        auto * mapped = ggml_dup(ctx, ids);
        ggml_set_name(mapped, ("moe_stream_ids." + std::to_string(il)).c_str());
        ggml_backend_sched_set_tensor_backend(sched, mapped, backend);
        return mapped;
    }
    return ids;
}
