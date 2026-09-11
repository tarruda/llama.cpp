#include "models.h"
#include "llama-memory-dsv41.h"
#include "../../ggml/src/ggml-quants.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

static void dsv41_require(bool valid, const char * message) {
    if (!valid) {
        throw std::runtime_error(std::string("DeepSeek-V4.1: ") + message);
    }
}

void llama_model_deepseek41::load_arch_hparams(llama_model_loader & ml) {
    dsv41_require(hparams.n_layer_nextn == 0, "draft layers require a separate model");
    std::string dense_dtype = "fp8", expert_dtype = "fp8";
    ml.get_key(LLM_KV_DENSE_ACTIVATION_DTYPE, dense_dtype, false);
    ml.get_key(LLM_KV_EXPERT_ACTIVATION_DTYPE, expert_dtype, false);
    dsv41_require((dense_dtype == "bf16" || dense_dtype == "fp8") && (expert_dtype == "bf16" || expert_dtype == "fp8"), "activation dtype must be bf16 or fp8");
    hparams.dsv41_dense_act_fp8 = dense_dtype == "fp8";
    hparams.dsv41_expert_act_fp8 = expert_dtype == "fp8";
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_ATTENTION_Q_LORA_RANK, hparams.n_lora_q);
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa);
    ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer_all);
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT, hparams.n_expert_shared);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE, hparams.expert_weights_scale);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM, hparams.expert_weights_norm);
    ml.get_key(LLM_KV_EXPERT_GATING_FUNC, hparams.expert_gating_func);
    ml.get_key_or_arr(LLM_KV_SWIGLU_CLAMP_EXP, hparams.swiglu_clamp_exp, hparams.n_layer_all);
    ml.get_key_or_arr(LLM_KV_SWIGLU_CLAMP_SHEXP, hparams.swiglu_clamp_shexp, hparams.n_layer_all);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, hparams.indexer_n_head);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, hparams.indexer_head_size);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_TOP_K, hparams.indexer_top_k);
    ml.get_key(LLM_KV_ATTENTION_OUTPUT_GROUP_COUNT, hparams.dsv4_o_group_count);
    ml.get_key(LLM_KV_ATTENTION_OUTPUT_LORA_RANK, hparams.dsv4_o_lora_rank);
    ml.get_key(LLM_KV_ATTENTION_COMPRESS_ROPE_FREQ_BASE, hparams.dsv4_compress_rope_base);
    ml.get_key(LLM_KV_HYPER_CONNECTION_COUNT, hparams.dsv4_hc_mult);
    ml.get_key(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS, hparams.dsv4_hc_sinkhorn_iters);
    ml.get_key(LLM_KV_HYPER_CONNECTION_EPSILON, hparams.dsv4_hc_eps);
    uint32_t candidate_source;
    ml.get_key(LLM_KV_ATTENTION_INDEXER_CANDIDATE_SOURCE_LAYER, candidate_source);
    dsv41_require(candidate_source <= INT32_MAX || candidate_source == UINT32_MAX, "invalid candidate source layer");
    hparams.dsv41_candidate_source_layer = candidate_source == UINT32_MAX ? -1 : (int32_t) candidate_source;
    ml.get_key(LLM_KV_ATTENTION_INDEXER_CANDIDATE_TOPK_BLOCKS, hparams.dsv41_candidate_topk_blocks);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_CANDIDATE_BLOCK_SIZE, hparams.dsv41_candidate_block_size);

    dsv41_require(hparams.n_embd > 0 && hparams.n_embd % 32 == 0 && hparams.n_lora_q > 0 && hparams.n_lora_q % 32 == 0, "invalid embedding or query rank");
    dsv41_require(hparams.dsv4_hc_mult == 4 && hparams.dsv4_hc_sinkhorn_iters > 0 && hparams.dsv4_hc_eps > 0, "invalid hyper-connection parameters");
    dsv41_require(hparams.f_norm_rms_eps > 0 && hparams.n_swa > 0, "invalid normalization epsilon or window size");
    dsv41_require(hparams.dsv4_o_group_count > 0 && hparams.dsv4_o_lora_rank > 0, "invalid output groups or rank");
    dsv41_require(hparams.indexer_n_head > 0 && hparams.indexer_head_size > 0 && hparams.indexer_head_size % 32 == 0 && hparams.indexer_top_k > 0, "invalid indexer dimensions");
    dsv41_require(hparams.n_expert > 0 && hparams.n_expert_shared == 1 && hparams.expert_weights_norm && hparams.expert_gating_func == LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS, "expected normalized sqrtsoftplus routing with one shared expert");

    std::vector<uint32_t> ratios, kv_sources, index_sources;
    ml.get_arr(LLM_KV_ATTENTION_COMPRESS_RATIOS, ratios);
    ml.get_arr(LLM_KV_ATTENTION_KV_SOURCE_LAYERS, kv_sources);
    ml.get_arr(LLM_KV_ATTENTION_INDEX_SOURCE_LAYERS, index_sources);
    dsv41_require(ratios.size() >= hparams.n_layer(), "compress_ratios is shorter than block_count");
    std::fill(hparams.dsv4_compress_ratios.begin(), hparams.dsv4_compress_ratios.end(), 0);
    std::copy_n(ratios.begin(), hparams.n_layer(), hparams.dsv4_compress_ratios.begin());

    const auto resolve_sources = [&](const std::vector<uint32_t> & sources, std::array<int32_t, LLAMA_MAX_LAYERS> & mapping) {
        dsv41_require(!sources.empty() && std::is_sorted(sources.begin(), sources.end()) && std::adjacent_find(sources.begin(), sources.end()) == sources.end() && sources.back() < hparams.n_layer(), "source layers must be distinct, increasing backbone indices");
        mapping.fill(-1);
        int32_t source = -1;
        size_t next = 0;
        for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
            if (next < sources.size() && sources[next] == il) {
                source = il;
                ++next;
                dsv41_require(ratios[il] != 0, "a source layer must have compressed KV");
            }
            if (ratios[il] != 0) {
                dsv41_require(source >= 0 && ratios[source] == ratios[il], "compressed layer has no source with the same ratio");
                mapping[il] = source;
            }
        }
    };
    resolve_sources(kv_sources, hparams.dsv41_kv_source);
    resolve_sources(index_sources, hparams.dsv41_index_source);
    for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
        dsv41_require(ratios[il] <= 2, "only compression ratios 0, 1, and 2 are supported");
        dsv41_require(hparams.n_head_kv(il) == 1 && hparams.n_head(il) % hparams.dsv4_o_group_count == 0, "invalid attention head layout");
        dsv41_require(hparams.n_embd_head_k(il) == hparams.n_embd_head_v(il) && hparams.n_embd_head_k(il) % 64 == 0 && hparams.n_rot(il) > 0 && hparams.n_rot(il) <= hparams.n_embd_head_k(il) && hparams.n_rot(il) % 2 == 0, "invalid latent or rotary dimensions");
        dsv41_require(hparams.n_ff_exp(il) > 0 && hparams.n_ff_exp(il) % 32 == 0, "invalid expert width");
        if (hparams.dsv41_kv_source[il] == (int32_t) il) {
            dsv41_require(hparams.dsv41_index_source[il] == (int32_t) il, "a KV source must also produce index keys");
        }
        if (ratios[il] != 0) {
            dsv41_require(hparams.dsv41_index_source[il] >= hparams.dsv41_kv_source[il], "index reuse crosses a KV source boundary");
        }
    }
    if (hparams.dsv41_candidate_source_layer >= 0) {
        const auto source = (uint32_t) hparams.dsv41_candidate_source_layer;
        dsv41_require(source < hparams.n_layer() && hparams.dsv41_kv_source[source] == (int32_t) source && ratios[source] == 1, "candidate source must produce ratio-1 KV");
        dsv41_require(hparams.dsv41_candidate_topk_blocks > 0 && hparams.dsv41_candidate_block_size > 0, "invalid candidate block limits");
    }

    ml.get_arr(LLM_KV_ENGRAM_LAYERS, engram.layers);
    ml.get_key(LLM_KV_ENGRAM_NGRAM_SIZE, engram.ngram_size);
    ml.get_key(LLM_KV_ENGRAM_HEAD_COUNT, engram.n_heads);
    ml.get_key(LLM_KV_ENGRAM_HEAD_DIM, engram.head_dim);
    ml.get_key(LLM_KV_ENGRAM_COMPRESSED_VOCAB_SIZE, engram.compressed_vocab_size);
    ml.get_key(LLM_KV_ENGRAM_PAD_TOKEN_ID, engram.pad_token_id);
    ml.get_arr(LLM_KV_ENGRAM_EMBEDDING_COUNTS, engram.embedding_counts);
    ml.get_arr(LLM_KV_ENGRAM_HEAD_BUCKET_SIZES, engram.bucket_sizes);
    ml.get_arr(LLM_KV_ENGRAM_HEAD_OFFSETS, engram.offsets);
    ml.get_arr(LLM_KV_ENGRAM_MULTIPLIERS, engram.multipliers);
    ml.get_arr(LLM_KV_ENGRAM_TOKEN_MAP, engram.token_map);
    dsv41_require(engram.ngram_size >= 2 && engram.ngram_size <= LLAMA_MAX_PLE_NGRAM && engram.n_heads > 0 && engram.head_dim > 0 && engram.head_dim % 32 == 0, "invalid Engram dimensions");
    dsv41_require(std::is_sorted(engram.layers.begin(), engram.layers.end()) && std::adjacent_find(engram.layers.begin(), engram.layers.end()) == engram.layers.end(), "Engram layers must be distinct and increasing");
    const size_t heads = (engram.ngram_size - 1)*engram.n_heads;
    dsv41_require(engram.embedding_counts.size() == engram.layers.size() && engram.bucket_sizes.size() == heads*engram.layers.size() && engram.offsets.size() == engram.bucket_sizes.size() && engram.multipliers.size() == engram.ngram_size*engram.layers.size(), "inconsistent Engram array lengths");
    dsv41_require(engram.pad_token_id < engram.token_map.size() && engram.compressed_vocab_size > 0, "invalid Engram padding or vocabulary");
    for (int32_t id : engram.token_map) {
        dsv41_require(id >= 0 && (uint32_t) id < engram.compressed_vocab_size, "Engram token map contains an invalid ID");
    }
    for (size_t i = 0; i < engram.layers.size(); ++i) {
        dsv41_require(engram.layers[i] < hparams.n_layer() && engram.embedding_counts[i] > 0 && engram.embedding_counts[i] <= INT32_MAX, "invalid Engram layer or row count");
        uint64_t offset = 0;
        for (size_t head = i*heads; head < (i + 1)*heads; ++head) {
            dsv41_require(engram.bucket_sizes[head] > 0 && engram.offsets[head] == offset && engram.bucket_sizes[head] <= engram.embedding_counts[i] - offset, "invalid Engram bucket range");
            offset += engram.bucket_sizes[head];
        }
        dsv41_require(offset == engram.embedding_counts[i], "Engram buckets do not cover the embedding table");
    }

    // The final HC collapse returns one embedding, not the four-copy residual stream.
    hparams.n_embd_out_impl = hparams.n_embd;
    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
    hparams.set_swa_pattern(0);
    type = LLM_TYPE_UNKNOWN;
}

void llama_model_deepseek41::load_arch_tensors(llama_model_loader & ml) {
    GGML_UNUSED(ml);
    LLAMA_LOAD_LOCALS;
    const int64_t hc = hparams.dsv4_hc_mult;
    const int64_t hc_dim = hc*n_embd;
    const int64_t mix_dim = (2 + hc)*hc;
    const int64_t q_rank = hparams.n_lora_q;
    const int64_t head_dim = hparams.n_embd_head_k();
    const int64_t groups = hparams.dsv4_o_group_count;
    const int64_t o_rank = hparams.dsv4_o_lora_rank;
    const int64_t index_dim = hparams.indexer_head_size;
    dsv41_require(engram.token_map.size() == (size_t) n_vocab, "Engram token map does not cover the tokenizer vocabulary");

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, 0);

    for (int il = 0; il < n_layer; ++il) {
        auto & layer = layers[il];
        const int64_t n_ff_exp = hparams.n_ff_exp(il);
        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", il), {n_embd}, 0);
        layer.attn_sinks = create_tensor(tn(LLM_TENSOR_ATTN_SINKS, "weight", il), {n_head}, 0);
        layer.wq_a = create_tensor(tn(LLM_TENSOR_ATTN_Q_A, "weight", il), {n_embd, q_rank}, 0);
        layer.attn_q_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_A_NORM, "weight", il), {q_rank}, 0);
        layer.wq_b = create_tensor(tn(LLM_TENSOR_ATTN_Q_B, "weight", il), {q_rank, n_head*head_dim}, 0);
        layer.wkv = create_tensor(tn(LLM_TENSOR_ATTN_KV, "weight", il), {n_embd, head_dim}, 0);
        layer.attn_kv_norm = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_NORM, "weight", il), {head_dim}, 0);
        layer.wo_a = create_tensor(tn(LLM_TENSOR_ATTN_OUT_A, "weight", il), {n_head*head_dim/groups, groups*o_rank}, 0);
        layer.wo_b = create_tensor(tn(LLM_TENSOR_ATTN_OUT_B, "weight", il), {groups*o_rank, n_embd}, 0);
        layer.hc_attn_fn = create_tensor(tn(LLM_TENSOR_HC_ATTN_FN, "weight", il), {hc_dim, mix_dim}, 0);
        layer.hc_attn_base = create_tensor(tn(LLM_TENSOR_HC_ATTN_BASE, "weight", il), {mix_dim}, 0);
        layer.hc_attn_scale = create_tensor(tn(LLM_TENSOR_HC_ATTN_SCALE, "weight", il), {3}, 0);
        layer.hc_ffn_fn = create_tensor(tn(LLM_TENSOR_HC_FFN_FN, "weight", il), {hc_dim, mix_dim}, 0);
        layer.hc_ffn_base = create_tensor(tn(LLM_TENSOR_HC_FFN_BASE, "weight", il), {mix_dim}, 0);
        layer.hc_ffn_scale = create_tensor(tn(LLM_TENSOR_HC_FFN_SCALE, "weight", il), {3}, 0);

        if (hparams.dsv41_kv_source[il] == il) {
            layer.attn_comp_wkv = create_tensor(tn(LLM_TENSOR_ATTN_COMPRESSOR_WKV, "weight", il), {n_embd, head_dim}, 0);
            layer.attn_comp_norm = create_tensor(tn(LLM_TENSOR_ATTN_COMPRESSOR_NORM, "weight", il), {head_dim}, 0);
            if (hparams.dsv4_compress_ratios[il] > 1) {
                layer.attn_comp_wgate = create_tensor(tn(LLM_TENSOR_ATTN_COMPRESSOR_WGATE, "weight", il), {n_embd, head_dim}, 0);
            }
            layer.indexer_attn_k = create_tensor(tn(LLM_TENSOR_INDEXER_ATTN_K, "weight", il), {head_dim, index_dim}, 0);
            layer.indexer_k_norm = create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM, "weight", il), {index_dim}, 0);
        }
        if (hparams.dsv41_index_source[il] == il) {
            layer.indexer_proj = create_tensor(tn(LLM_TENSOR_INDEXER_PROJ, "weight", il), {n_embd, hparams.indexer_n_head}, 0);
            layer.indexer_attn_q_b = create_tensor(tn(LLM_TENSOR_INDEXER_ATTN_Q_B, "weight", il), {q_rank, hparams.indexer_n_head*index_dim}, 0);
        }
        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", il), {n_embd}, 0);
        layer.ffn_gate_inp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", il), {n_embd, n_expert}, 0);
        layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", il), {n_expert}, 0);
        layer.ffn_exp_probs_b_vl = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B_VL, "bias", il), {n_expert}, TENSOR_NOT_REQUIRED);
        layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", il), {n_embd, n_ff_exp, n_expert}, 0);
        layer.ffn_up_exps = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS, "weight", il), {n_embd, n_ff_exp, n_expert}, 0);
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", il), {n_ff_exp, n_embd, n_expert}, 0);
        layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", il), {n_embd, n_ff_exp}, 0);
        layer.ffn_up_shexp = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP, "weight", il), {n_embd, n_ff_exp}, 0);
        layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", il), {n_ff_exp, n_embd}, 0);
    }
    for (size_t i = 0; i < engram.layers.size(); ++i) {
        const int il = engram.layers[i];
        auto & layer = layers[il];
        const int64_t rows = engram.embedding_counts[i];
        const int64_t dim = engram.head_dim;
        const int64_t embedding_dim = (engram.ngram_size - 1)*engram.n_heads*dim;
        layer.engram_embd = create_tensor(tn(LLM_TENSOR_ENGRAM_EMBD, "weight", il), {dim, rows}, TENSOR_READ_LAZY);
        layer.engram_embd_scale = create_tensor(tn(LLM_TENSOR_ENGRAM_EMBD_SCALE, "weight", il), {dim/32, rows}, TENSOR_READ_LAZY);
        dsv41_require(layer.engram_embd->type == GGML_TYPE_I8 && layer.engram_embd_scale->type == GGML_TYPE_I8, "Engram tables must retain FP8 bytes");
        layer.engram_kv = create_tensor(tn(LLM_TENSOR_ENGRAM_KV, "weight", il), {embedding_dim, (hc + 1)*n_embd}, 0);
        layer.engram_k_weight = create_tensor(tn(LLM_TENSOR_ENGRAM_K_WEIGHT, "weight", il), {n_embd, hc}, 0);
        layer.engram_q_weight = create_tensor(tn(LLM_TENSOR_ENGRAM_Q_WEIGHT, "weight", il), {n_embd, hc}, 0);
    }
}

namespace {

class llm_graph_input_dsv41 : public llm_graph_input_i {
public:
    llm_graph_input_dsv41(const llama_model_deepseek41 & model, llama_memory_dsv41 * memory) : model(model), memory(memory) {}

    void init_rope(int n_rot, float base, float comp_base, float scale, float ext, int original_context, float beta_fast, float beta_slow) {
        for (int mode = 0; mode < 2; ++mode) {
            auto & freq = frequencies[mode];
            freq.resize(n_rot/2);
            const float theta = mode ? comp_base : base;
            const auto corrected = [&](double rotations) {
                return n_rot*std::log(original_context/(rotations*2*3.14159265358979323846))/(2*std::log(double(theta)));
            };
            const float low = mode && ext != 0 ? std::max(0.0, std::floor(corrected(beta_fast))) : 0;
            const float high = mode && ext != 0 ? std::min(double(n_rot - 1), std::ceil(corrected(beta_slow))) : 0;
            for (int i = 0; i < n_rot/2; ++i) {
                float value = 1.0f/std::pow(theta, float(2*i)/n_rot);
                if (mode) {
                    const float smooth = ext*(1.0f - std::clamp((i - low)/std::max(high - low, 1e-3f), 0.0f, 1.0f));
                    value = value*scale*(1.0f - smooth) + value*smooth;
                }
                freq[i] = value;
            }
        }
    }

    void set_input(const llama_ubatch * ubatch) override {
        const auto & e = model.engram;
        const uint32_t heads = (e.ngram_size - 1)*e.n_heads;
        std::vector<int32_t> pos(n_tokens);
        for (int64_t it = 0; it < n_tokens; ++it) {
            pos[it] = replay ? memory->tokens.size() - n_tokens + it : ubatch->pos[it];
        }
        if (positions) { ggml_backend_tensor_set(positions, pos.data(), 0, pos.size()*sizeof(int32_t)); }
        const int32_t window_start = replay ? pos[0] : memory->decoder_start;
        for (auto * attention : decoder_attention) { ggml_dsv41_attn_set_window_start(attention, window_start); }
        for (int ratio = 0; ratio < 3; ++ratio) {
            if (!cache_rows[ratio]) { continue; }
            std::vector<int32_t> ids(n_tokens);
            for (int64_t it = 0; it < n_tokens; ++it) {
                ids[it] = ratio == 0 ? pos[it] % memory->n_ring : (pos[it] + 1) % ratio == 0 ? pos[it]/ratio : -1;
            }
            ggml_backend_tensor_set(cache_rows[ratio], ids.data(), 0, ids.size()*sizeof(int32_t));
        }
        for (int mode = 0; mode < 3; ++mode) {
            if (!rotations[mode]) { continue; }
            const auto & freq = frequencies[mode != 0];
            std::vector<float> cs(n_tokens*freq.size()*2);
            for (int64_t it = 0; it < n_tokens; ++it) {
                const int32_t position = mode == 2 ? pos[it]/2*2 : pos[it];
                for (size_t i = 0; i < freq.size(); ++i) {
                    const float angle = position*freq[i];
                    cs[2*(it*freq.size() + i)]     = std::cos(angle);
                    cs[2*(it*freq.size() + i) + 1] = std::sin(angle);
                }
            }
            ggml_backend_tensor_set(rotations[mode], cs.data(), 0, cs.size()*sizeof(float));
        }
        std::vector<float> values(engram.empty() ? 0 : n_tokens*heads*e.head_dim);
        for (size_t ie = 0; ie < engram.size(); ++ie) {
            const auto & layer = model.layers[e.layers[ie]];
            for (int64_t it = 0; it < n_tokens; ++it) {
                uint64_t hash = 0;
                for (uint32_t shift = 0; shift < e.ngram_size; ++shift) {
                    const int64_t pos = (int64_t) ubatch->pos[it] - shift;
                    const auto token = pos < 0 ? e.pad_token_id : (uint32_t) memory->tokens.at(pos);
                    hash ^= (uint64_t) e.token_map[token]*e.multipliers[ie*e.ngram_size + shift];
                    if (shift == 0) {
                        continue;
                    }
                    for (uint32_t ih = 0; ih < e.n_heads; ++ih) {
                        const auto col = (shift - 1)*e.n_heads + ih;
                        const auto bucket = ie*heads + col;
                        const auto row = hash % e.bucket_sizes[bucket] + e.offsets[bucket];
                        uint8_t packed[33];
                        for (uint32_t block = 0; block < e.head_dim/32; ++block) {
                            ggml_backend_tensor_get(layer.engram_embd_scale, packed, row*layer.engram_embd_scale->nb[1] + block, 1);
                            ggml_backend_tensor_get(layer.engram_embd, packed + 1, row*layer.engram_embd->nb[1] + block*32, 32);
                            dequantize_row_mxfp8_act(packed, values.data() + (it*heads + col)*e.head_dim + block*32, 32);
                        }
                    }
                }
            }
            ggml_backend_tensor_set(engram[ie], values.data(), 0, values.size()*sizeof(float));
        }
    }

    bool can_reuse(const llm_graph_params & params) override {
        const auto * mctx = static_cast<const llama_memory_dsv41_context *>(params.mctx);
        const int64_t count = replay ? std::min<size_t>(memory->tokens.size(), model.hparams.n_swa) : params.ubatch.n_tokens;
        return mctx && mctx->memory == memory && n_tokens == count;
    }

    const llama_model_deepseek41 & model;
    llama_memory_dsv41 * memory;
    int64_t n_tokens = 0;
    bool replay = false;
    ggml_tensor * positions = nullptr;
    std::vector<ggml_tensor *> decoder_attention;
    std::array<std::vector<float>, 2> frequencies;
    std::array<ggml_tensor *, 3> rotations = {};
    std::array<ggml_tensor *, 3> cache_rows = {};
    std::vector<ggml_tensor *> engram;
};

struct dsv41_graph : public llm_graph_context {
    const llama_model_deepseek41 & model;
    llm_graph_input_dsv41 * input;
    ggml_tensor * positions;
    int64_t tokens;
    bool replay = false;
    std::vector<ggml_tensor *> cache_kv;
    std::vector<ggml_tensor *> cache_index;
    std::vector<ggml_tensor *> indices;
    ggml_tensor * candidates = nullptr;

    ggml_tensor * bf16(ggml_tensor * x) const {
        return ggml_dsv41_act_quant(ctx0, x, GGML_DSV41_QUANT_BF16);
    }

    ggml_tensor * linear(ggml_tensor * w, ggml_tensor * x, bool fp8) const {
        if (fp8) { x = ggml_dsv41_act_quant(ctx0, x, GGML_DSV41_QUANT_MXFP8); }
        auto * out = build_lora_mm(w, x);
        if (!fp8) {
            ggml_prec_set_src(out, GGML_PREC_F32, 0);
            ggml_prec_set_src(out, GGML_PREC_F32, 1);
        }
        return bf16(out);
    }

    ggml_tensor * norm(ggml_tensor * x, ggml_tensor * weight) const {
        if (weight->type != GGML_TYPE_F32) { weight = ggml_cast(ctx0, weight, GGML_TYPE_F32); }
        return bf16(ggml_mul(ctx0, ggml_rms_norm(ctx0, x, norm_rms_eps), weight));
    }

    ggml_tensor * rstd(ggml_tensor * x) const {
        auto * variance = ggml_scale_bias(ctx0, ggml_mean(ctx0, ggml_sqr(ctx0, x)), 1.0f, norm_rms_eps);
        auto * one = ggml_fill(ctx0, ggml_dup_tensor(ctx0, variance), 1.0f);
        return ggml_div(ctx0, one, ggml_sqrt(ctx0, variance));
    }

    ggml_tensor * mm_f32(ggml_tensor * weight, ggml_tensor * x) const {
        auto * out = ggml_mul_mat(ctx0, weight, x);
        ggml_prec_set_src(out, GGML_PREC_F32, 0);
        ggml_prec_set_src(out, GGML_PREC_F32, 1);
        return out;
    }

    struct hc_mixes {
        ggml_tensor * pre;
        ggml_tensor * post;
        ggml_tensor * comb;
    };

    hc_mixes mix(ggml_tensor * x, ggml_tensor * fn, ggml_tensor * scale, ggml_tensor * base, int il, bool ffn) const {
        const int64_t hc = hparams.dsv4_hc_mult;
        auto * flat = ggml_reshape_2d(ctx0, x, hc*n_embd, tokens);
        auto * mixes = ggml_mul(ctx0, mm_f32(fn, flat), rstd(flat));
        cb(mixes, ffn ? "dsv41_ffn_mixes" : "dsv41_attn_mixes", il);
        auto * split = ggml_dsv41_hc_split(ctx0, mixes, scale, base, hparams.dsv4_hc_eps, hparams.dsv4_hc_sinkhorn_iters);
        hc_mixes result = {
            ggml_view_2d(ctx0, split, hc, tokens, split->nb[1], 0),
            ggml_view_2d(ctx0, split, hc, tokens, split->nb[1], hc*sizeof(float)),
            ggml_view_3d(ctx0, split, hc, hc, tokens, hc*sizeof(float), split->nb[1], 2*hc*sizeof(float)),
        };
        cb(result.pre, ffn ? "dsv41_ffn_pre_mix" : "dsv41_attn_pre_mix", il);
        cb(result.post, ffn ? "dsv41_ffn_post_mix" : "dsv41_attn_post_mix", il);
        cb(result.comb, ffn ? "dsv41_ffn_comb" : "dsv41_attn_comb", il);
        return result;
    }

    ggml_tensor * rope(ggml_tensor * x, int il, bool inverse = false, bool grouped = false) const {
        const int mode = grouped ? 2 : hparams.dsv4_compress_ratios[il] != 0;
        auto * & rotations = input->rotations[mode];
        if (!rotations) {
            rotations = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_rot, tokens);
            ggml_set_input(rotations);
        }
        return ggml_dsv41_rope(ctx0, x, rotations, inverse);
    }

    ggml_tensor * hc_post(ggml_tensor * x, ggml_tensor * residual, const hc_mixes & mix) const {
        const int64_t hc = hparams.dsv4_hc_mult;
        ggml_tensor * out = nullptr;
        for (int64_t dst = 0; dst < hc; ++dst) {
            ggml_tensor * sum = nullptr;
            for (int64_t src = 0; src < hc; ++src) {
                auto * row = ggml_view_2d(ctx0, residual, n_embd, tokens, residual->nb[2], src*residual->nb[1]);
                auto * weight = ggml_view_2d(ctx0, mix.comb, 1, tokens, mix.comb->nb[2], dst*mix.comb->nb[0] + src*mix.comb->nb[1]);
                auto * term = ggml_mul(ctx0, row, weight);
                sum = sum ? ggml_add(ctx0, sum, term) : term;
            }
            auto * post = ggml_view_2d(ctx0, mix.post, 1, tokens, mix.post->nb[1], dst*mix.post->nb[0]);
            auto * copy = ggml_reshape_3d(ctx0, ggml_add(ctx0, ggml_mul(ctx0, x, post), sum), n_embd, 1, tokens);
            out = out ? ggml_concat(ctx0, out, copy, 1) : copy;
        }
        return bf16(out);
    }

    ggml_tensor * write_cache(ggml_tensor * cache, ggml_tensor * x, int ratio, ggml_dsv41_quant_type type) const {
        auto * & rows = input->cache_rows[ratio];
        if (!rows) {
            rows = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, tokens);
            ggml_set_input(rows);
        }
        return ggml_dsv41_set_rows(ctx0, cache, x, rows, type);
    }

    ggml_tensor * engram(ggml_tensor * x, size_t ie) const {
        const int il = model.engram.layers[ie];
        const auto & layer = model.layers[il];
        const int64_t hc = hparams.dsv4_hc_mult;
        auto * embedding = bf16(input->engram[ie]);
        cb(embedding, "dsv41_engram_embedding", il);
        auto * kv = linear(layer.engram_kv, embedding, hparams.dsv41_dense_act_fp8);
        cb(kv, "dsv41_engram_kv", il);
        auto * key = ggml_view_3d(ctx0, kv, n_embd, hc, tokens, n_embd*sizeof(float), kv->nb[1], 0);
        auto * value = ggml_view_2d(ctx0, kv, n_embd, tokens, kv->nb[1], hc*n_embd*sizeof(float));
        auto * weight = ggml_mul(ctx0, ggml_cast(ctx0, layer.engram_q_weight, GGML_TYPE_F32), ggml_cast(ctx0, layer.engram_k_weight, GGML_TYPE_F32));
        auto * out = bf16(ggml_dsv41_engram(ctx0, x, key, value, weight, norm_rms_eps, 1e-6f));
        cb(out, "dsv41_engram", il);
        return out;
    }

    void publish_context(ggml_tensor * x, int il) {
        const auto & layer = model.layers[il];
        const auto & cache = input->memory->layers[il];
        const int64_t dim = hparams.n_embd_head_k(il);
        const int ratio = hparams.dsv4_compress_ratios[il];
        ggml_tensor * latent;
        if (ratio == 2) {
            auto * kv = mm_f32(layer.attn_comp_wkv, x);
            auto * score = mm_f32(layer.attn_comp_wgate, x);
            kv = ggml_set_rows(ctx0, cache.comp_kv, kv, input->cache_rows[0]);
            score = ggml_set_rows(ctx0, cache.comp_score, score, input->cache_rows[0]);
            latent = ggml_dsv41_pool(ctx0, kv, score, positions);
            if (ggml_backend_buffer_is_host(cache.comp_kv->buffer)) {
                ggml_backend_sched_set_tensor_backend(sched, latent, backend_cpu);
            }
        } else {
            latent = linear(layer.attn_comp_wkv, x, false);
        }
        latent = norm(latent, layer.attn_comp_norm);
        cb(latent, "dsv41_latent", il);
        auto * key = norm(linear(layer.indexer_attn_k, latent, false), layer.indexer_k_norm);
        key = rope(ggml_reshape_3d(ctx0, key, hparams.indexer_head_size, 1, tokens), il, false, ratio == 2);
        key = ggml_reshape_2d(ctx0, key, hparams.indexer_head_size, tokens);
        cb(key, "dsv41_index_key", il);
        auto * kv = rope(ggml_reshape_3d(ctx0, latent, dim, 1, tokens), il, false, ratio == 2);
        kv = ggml_reshape_2d(ctx0, kv, dim, tokens);
        cache_kv[il] = write_cache(cache.kv, kv, ratio, GGML_DSV41_QUANT_NVFP4);
        cache_index[il] = write_cache(cache.index, key, ratio, GGML_DSV41_QUANT_MXFP4);
    }

    ggml_tensor * attention(ggml_tensor * x, int il) {
        const auto & layer = model.layers[il];
        const bool fp8 = hparams.dsv41_dense_act_fp8;
        const auto & cache = input->memory->layers[il];
        const int64_t dim = hparams.n_embd_head_k(il), heads = hparams.n_head(il);
        const int ratio = hparams.dsv4_compress_ratios[il];
        auto * qa = linear(layer.wq_a, x, fp8);
        cb(qa, "dsv41_q_a", il);
        auto * qr = norm(qa, layer.attn_q_a_norm);
        cb(qr, "dsv41_q_norm", il);
        auto * q = linear(layer.wq_b, qr, fp8);
        cb(q, "dsv41_q_b", il);
        q = rope(ggml_reshape_3d(ctx0, q, dim, heads, tokens), il);
        cb(q, "dsv41_q", il);
        auto * raw = norm(linear(layer.wkv, x, fp8), layer.attn_kv_norm);
        raw = rope(ggml_reshape_3d(ctx0, raw, dim, 1, tokens), il);
        raw = ggml_reshape_2d(ctx0, raw, dim, tokens);
        cb(raw, "dsv41_raw_kv", il);
        raw = write_cache(cache.raw, raw, 0, GGML_DSV41_QUANT_MXFP8);
        if (!replay && hparams.dsv41_kv_source[il] == il) { publish_context(x, il); }
        ggml_tensor * selected = nullptr;
        if (ratio) {
            if (hparams.dsv41_index_source[il] == il) {
                auto * qi = linear(layer.indexer_attn_q_b, qr, fp8);
                qi = rope(ggml_reshape_3d(ctx0, qi, hparams.indexer_head_size, hparams.indexer_n_head, tokens), il);
                qi = ggml_dsv41_act_quant(ctx0, qi, GGML_DSV41_QUANT_MXFP4);
                cb(qi, "dsv41_index_query", il);
                auto * weights = linear(layer.indexer_proj, x, false);
                weights = bf16(ggml_scale(ctx0, weights, 1.0f/std::sqrt(float(hparams.indexer_head_size*hparams.indexer_n_head))));
                cb(weights, "dsv41_index_weights", il);
                auto * keys = cache_index[hparams.dsv41_kv_source[il]];
                ggml_tensor * blocks = candidates ? ggml_view_2d(ctx0, candidates, hparams.dsv41_candidate_topk_blocks, tokens, candidates->nb[1], hparams.indexer_top_k*sizeof(int32_t)) : nullptr;
                auto * scores = ggml_dsv41_index_scores(ctx0, qi, keys, weights, positions, blocks, ratio, hparams.dsv41_candidate_block_size);
                const bool on_cpu = ggml_backend_buffer_is_host(input->memory->layers[hparams.dsv41_kv_source[il]].index->buffer);
                if (on_cpu) {
                    ggml_backend_sched_set_tensor_backend(sched, scores, backend_cpu);
                }
                cb(scores, "dsv41_index_scores", il);
                const bool source = il == hparams.dsv41_candidate_source_layer;
                indices[il] = ggml_dsv41_select(ctx0, scores, positions, blocks, hparams.indexer_top_k, ratio,
                        source ? hparams.dsv41_candidate_topk_blocks : 0, hparams.dsv41_candidate_block_size, source);
                if (on_cpu) { ggml_backend_sched_set_tensor_backend(sched, indices[il], backend_cpu); }
                cb(indices[il], "dsv41_index_selected", il);
                if (source) {
                    candidates = indices[il];
                    indices[il] = ggml_cont(ctx0, ggml_view_2d(ctx0, candidates, hparams.indexer_top_k, tokens, candidates->nb[1], 0));
                }
            }
            selected = indices[hparams.dsv41_index_source[il]];
        }
        auto * kv = ratio ? cache_kv[hparams.dsv41_kv_source[il]] : nullptr;
        auto * out = ggml_dsv41_attn(ctx0, q, raw, layer.attn_sinks, positions, selected, kv, hparams.n_swa, ratio);
        if (il >= n_layer/2) { input->decoder_attention.push_back(out); }
        if (ggml_backend_buffer_is_host(cache.raw->buffer) || (kv && ggml_backend_buffer_is_host(input->memory->layers[hparams.dsv41_kv_source[il]].kv->buffer))) {
            ggml_backend_sched_set_tensor_backend(sched, out, backend_cpu);
        }
        cb(out, "dsv41_sparse_attention", il);
        out = rope(out, il, true);
        cb(out, "dsv41_derope", il);
        const int64_t groups = hparams.dsv4_o_group_count, rank = hparams.dsv4_o_lora_rank;
        out = ggml_permute(ctx0, ggml_reshape_3d(ctx0, out, heads*dim/groups, groups, tokens), 0, 2, 1, 3);
        auto * wa = ggml_reshape_3d(ctx0, layer.wo_a, heads*dim/groups, rank, groups);
        auto * oa = bf16(mm_f32(wa, out));
        oa = ggml_cont_2d(ctx0, ggml_permute(ctx0, oa, 0, 2, 1, 3), rank*groups, tokens);
        cb(oa, "dsv41_o_a", il);
        out = linear(layer.wo_b, oa, fp8);
        cb(out, "dsv41_attention", il);
        return out;
    }

    ggml_tensor * moe(ggml_tensor * x, int il) const {
        const auto & layer = model.layers[il];
        const int64_t used = hparams.n_expert_used(il);
        auto * scores = ggml_sqrt(ctx0, ggml_softplus(ctx0, mm_f32(layer.ffn_gate_inp, x)));
        auto * ids = ggml_top_k(ctx0, ggml_add(ctx0, scores, layer.ffn_exp_probs_b), used);
        cb(ids, "dsv41_expert_ids", il);
        auto * weights = ggml_get_rows(ctx0, ggml_reshape_3d(ctx0, scores, 1, n_expert, tokens), ids);
        weights = ggml_reshape_2d(ctx0, weights, used, tokens);
        if (used > 1) {
            weights = ggml_div(ctx0, weights, ggml_scale_bias(ctx0, ggml_sum_rows(ctx0, weights), 1.0f, 1e-20f));
        }
        weights = ggml_scale(ctx0, weights, hparams.expert_weights_scale);
        cb(weights, "dsv41_expert_weights", il);
        weights = ggml_reshape_3d(ctx0, weights, 1, used, tokens);
        auto * cur = hparams.dsv41_expert_act_fp8 ? ggml_dsv41_act_quant(ctx0, x, GGML_DSV41_QUANT_MXFP8) : x;
        cur = ggml_reshape_3d(ctx0, cur, n_embd, 1, tokens);
        auto * gate = bf16(build_lora_mm_id(layer.ffn_gate_exps, cur, ids));
        auto * up = bf16(build_lora_mm_id(layer.ffn_up_exps, cur, ids));
        cb(gate, "dsv41_expert_gate", il);
        cb(up, "dsv41_expert_up", il);
        cur = ggml_dsv41_swiglu(ctx0, gate, up, weights, hparams.swiglu_clamp_exp[il]);
        cb(cur, "dsv41_expert_hidden", il);
        if (hparams.dsv41_expert_act_fp8) { cur = ggml_dsv41_act_quant(ctx0, cur, GGML_DSV41_QUANT_MXFP8); }
        cb(cur, "dsv41_expert_down_input", il);
        auto * experts = bf16(build_lora_mm_id(layer.ffn_down_exps, cur, ids));
        cb(experts, "dsv41_expert_output", il);
        ggml_tensor * sum = nullptr;
        for (int64_t i = 0; i < used; ++i) {
            auto * expert = ggml_view_2d(ctx0, experts, n_embd, tokens, experts->nb[2], i*experts->nb[1]);
            sum = sum ? ggml_add(ctx0, sum, expert) : expert;
        }
        const bool fp8 = hparams.dsv41_dense_act_fp8;
        gate = linear(layer.ffn_gate_shexp, x, fp8);
        up = linear(layer.ffn_up_shexp, x, fp8);
        cur = ggml_dsv41_swiglu(ctx0, gate, up, nullptr, hparams.swiglu_clamp_shexp[il]);
        auto * shared = linear(layer.ffn_down_shexp, cur, fp8);
        cb(shared, "dsv41_shared_expert", il);
        cur = bf16(ggml_add(ctx0, sum, shared));
        cb(cur, "dsv41_moe", il);
        return cur;
    }

    dsv41_graph(const llama_model_deepseek41 & model, const llm_graph_params & params) :
        llm_graph_context(params), model(model), tokens(n_tokens), cache_kv(n_layer), cache_index(n_layer), indices(n_layer) {
        const auto * memory_context = dynamic_cast<const llama_memory_dsv41_context *>(mctx);
        dsv41_require(memory_context != nullptr, "missing compact state context");
        auto * memory = memory_context->memory;
        const bool ced = params.gtype == LLM_GRAPH_TYPE_CED_PREFILL;
        const int split = n_layer/2;
        const auto & e = model.engram;
        if (ced) {
            dsv41_require(n_layer % 2 == 0 && n_outputs <= 1, "invalid CED layer or output count");
            for (int il = split; il < n_layer; ++il) {
                dsv41_require(hparams.dsv4_compress_ratios[il] == 1 && hparams.dsv41_kv_source[il] == split, "CED requires one shared decoder global cache");
            }
            for (auto il : e.layers) { dsv41_require(il < (uint32_t) split, "CED requires encoder-only Engram layers"); }
        }
        const auto add_input = [&]() {
            auto owner = std::make_unique<llm_graph_input_dsv41>(model, memory);
            input = owner.get();
            input->n_tokens = tokens;
            input->replay = replay;
            input->init_rope(n_rot, freq_base, hparams.dsv4_compress_rope_base, freq_scale, ext_factor, n_ctx_orig, beta_fast, beta_slow);
            res->add_input(std::move(owner));
        };
        add_input();
        for (size_t ie = 0; ie < e.layers.size(); ++ie) {
            auto * rows = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, (e.ngram_size - 1)*e.n_heads*e.head_dim, n_tokens);
            ggml_set_input(rows);
            input->engram.push_back(rows);
        }
        positions = build_inp_pos();
        auto * out_ids = ced ? nullptr : build_inp_out_ids();
        const int64_t hc = hparams.dsv4_hc_mult;
        auto * x = build_inp_embd(model.tok_embd);
        x = ggml_repeat_4d(ctx0, ggml_reshape_3d(ctx0, x, n_embd, 1, n_tokens), n_embd, hc, n_tokens, 1);
        ggml_tensor * pre = nullptr;
        size_t ie = 0;
        for (int il = 0; il < n_layer; ++il) {
            if (il == split) {
                auto * hidden = ggml_set_rows(ctx0, memory->encoder_hidden, ggml_reshape_2d(ctx0, x, hc*n_embd, tokens), input->cache_rows[0]);
                auto * mixes = ggml_set_rows(ctx0, memory->encoder_pre, pre, input->cache_rows[0]);
                ggml_build_forward_expand(gf, hidden);
                ggml_build_forward_expand(gf, mixes);
                if (ced) {
                    auto * boundary = norm(bf16(ggml_dsv4_hc_pre(ctx0, x, pre)), model.layers[split].attn_norm);
                    cb(boundary, "dsv41_decoder_global_input", split);
                    publish_context(boundary, split);
                    ggml_build_forward_expand(gf, cache_kv[split]);
                    ggml_build_forward_expand(gf, cache_index[split]);
                    if (n_outputs == 0) { return; }
                    tokens = memory_context->get_replay_tokens();
                    replay = true;
                    add_input();
                    positions = input->positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, tokens);
                    ggml_set_input(positions);
                    auto * rows = input->cache_rows[0] = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, tokens);
                    ggml_set_input(rows);
                    x = ggml_reshape_3d(ctx0, ggml_get_rows(ctx0, hidden, rows), n_embd, hc, tokens);
                    pre = ggml_get_rows(ctx0, mixes, rows);
                    cb(x, "dsv41_encoder_tail", split);
                    cb(pre, "dsv41_encoder_tail_pre", split);
                }
            }
            const auto & layer = model.layers[il];
            if (ie < e.layers.size() && e.layers[ie] == (uint32_t) il) { x = engram(x, ie++); }
            const auto attn_mix = mix(x, layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base, il, false);
            auto * collapsed = pre ? bf16(ggml_dsv4_hc_pre(ctx0, x, pre)) : ggml_cont(ctx0, ggml_view_2d(ctx0, x, n_embd, tokens, x->nb[2], 0));
            auto * cur = norm(collapsed, layer.attn_norm);
            cb(cur, "dsv41_attn_norm", il);
            cur = attention(cur, il);
            x = hc_post(cur, x, attn_mix);
            cb(x, "dsv41_attn_post", il);
            const auto ffn_mix = mix(x, layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base, il, true);
            cur = norm(bf16(ggml_dsv4_hc_pre(ctx0, x, attn_mix.pre)), layer.ffn_norm);
            cb(cur, "dsv41_ffn_norm", il);
            cur = moe(cur, il);
            x = hc_post(cur, x, ffn_mix);
            cb(x, "dsv41_layer", il);
            pre = ffn_mix.pre;
        }
        x = bf16(ggml_dsv4_hc_pre(ctx0, x, pre));
        x = norm(x, model.output_norm);
        if (ced) { x = ggml_cont(ctx0, ggml_view_2d(ctx0, x, n_embd, 1, x->nb[1], (tokens - 1)*x->nb[1])); }
        if (out_ids) { x = ggml_get_rows(ctx0, x, out_ids); }
        cb(x, "result_norm", -1);
        res->t_embd = x;
        res->t_logits = mm_f32(model.output, x);
        cb(res->t_logits, "result_output", -1);
        ggml_build_forward_expand(gf, res->t_logits);
    }
};

} // namespace

std::unique_ptr<llm_graph_context> llama_model_deepseek41::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<dsv41_graph>(*this, params);
}
