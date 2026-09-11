#include "models.h"
#include "llama-memory-dsv41.h"
#include "../../ggml/src/ggml-quants.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
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

float dsv41_bf16(float x) {
    return ggml_bf16_to_fp32(ggml_fp32_to_bf16(x));
}

const float * dsv41_frow(const ggml_tensor * t, int64_t row) {
    GGML_ASSERT(t->type == GGML_TYPE_F32 && t->nb[0] == sizeof(float));
    return (const float *) ((const char *) t->data + row*t->nb[1]);
}

struct dsv41_op_data {
    llama_memory_dsv41 * memory;
    int il;
};

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
        std::vector<float> values(ubatch->n_tokens*heads*e.head_dim);
        for (int mode = 0; mode < 3; ++mode) {
            if (!rotations[mode]) { continue; }
            const auto & freq = frequencies[mode != 0];
            std::vector<float> cs(ubatch->n_tokens*freq.size()*2);
            for (uint32_t it = 0; it < ubatch->n_tokens; ++it) {
                const int32_t pos = mode == 2 ? ubatch->pos[it]/2*2 : ubatch->pos[it];
                for (size_t i = 0; i < freq.size(); ++i) {
                    const float angle = pos*freq[i];
                    cs[2*(it*freq.size() + i)]     = std::cos(angle);
                    cs[2*(it*freq.size() + i) + 1] = std::sin(angle);
                }
            }
            ggml_backend_tensor_set(rotations[mode], cs.data(), 0, cs.size()*sizeof(float));
        }
        for (size_t ie = 0; ie < e.layers.size(); ++ie) {
            const auto & layer = model.layers[e.layers[ie]];
            for (uint32_t it = 0; it < ubatch->n_tokens; ++it) {
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
        return mctx && mctx->memory == memory && n_tokens == params.ubatch.n_tokens;
    }

    dsv41_op_data * add_op(int il) {
        ops.push_back(std::make_unique<dsv41_op_data>(dsv41_op_data{memory, il}));
        return ops.back().get();
    }

    const llama_model_deepseek41 & model;
    llama_memory_dsv41 * memory;
    int64_t n_tokens = 0;
    std::array<std::vector<float>, 2> frequencies;
    std::array<ggml_tensor *, 3> rotations = {};
    std::vector<ggml_tensor *> engram;
    std::vector<std::unique_ptr<dsv41_op_data>> ops;
};

// CPU cache operations establish the reference graph before backend-specific cache kernels.
void dsv41_pool(ggml_tensor * dst, int ith, int nth, void * userdata) {
    GGML_UNUSED(ith);
    GGML_UNUSED(nth);
    const auto & op = *(const dsv41_op_data *) userdata;
    auto & layer = op.memory->layers[op.il];
    const auto * pos = (const int32_t *) dst->src[2]->data;
    for (int64_t it = 0; it < dst->ne[1]; ++it) {
        const auto slot = pos[it] % op.memory->n_ring;
        const auto * kv = dsv41_frow(dst->src[0], it);
        const auto * score = dsv41_frow(dst->src[1], it);
        auto * saved_kv = (float *) ((char *) layer.comp_kv->data + slot*layer.comp_kv->nb[1]);
        auto * saved_score = (float *) ((char *) layer.comp_score->data + slot*layer.comp_score->nb[1]);
        std::memcpy(saved_kv, kv, dst->nb[1]);
        std::memcpy(saved_score, score, dst->nb[1]);
        auto * out = (float *) ((char *) dst->data + it*dst->nb[1]);
        for (int64_t j = 0; j < dst->ne[0]; ++j) {
            out[j] = 0;
            if (pos[it] % 2 == 0) {
                continue;
            }
            const auto prev = (pos[it] - 1) % op.memory->n_ring;
            const auto a = dsv41_frow(layer.comp_score, prev)[j];
            const auto b = score[j];
            const float maximum = std::max(a, b);
            const float ea = std::exp(a - maximum), eb = std::exp(b - maximum);
            out[j] = dsv41_bf16(dsv41_frow(layer.comp_kv, prev)[j]*(ea/(ea + eb)) + kv[j]*(eb/(ea + eb)));
        }
    }
}

void dsv41_publish(ggml_tensor * dst, int ith, int nth, void * userdata) {
    GGML_UNUSED(ith);
    GGML_UNUSED(nth);
    const auto & op = *(const dsv41_op_data *) userdata;
    const auto * pos = (const int32_t *) dst->src[2]->data;
    const auto ratio = op.memory->hparams.dsv4_compress_ratios[op.il];
    for (int64_t it = 0; it < dst->src[0]->ne[1]; ++it) {
        if ((pos[it] + 1) % ratio == 0) {
            op.memory->write_kv(op.il, pos[it]/ratio, dsv41_frow(dst->src[0], it));
            op.memory->write_index(op.il, pos[it]/ratio, dsv41_frow(dst->src[1], it));
        }
    }
    *(float *) dst->data = 0;
}

std::vector<int32_t> dsv41_topk(const std::vector<float> & scores, int count) {
    std::vector<int32_t> ids(scores.size());
    std::iota(ids.begin(), ids.end(), 0);
    count = std::min<int>(count, ids.size());
    // Keep tied selections independent of masked padding.
    const auto compare = [&](int32_t a, int32_t b) { return scores[a] > scores[b] || (scores[a] == scores[b] && a < b); };
    if (count > 0) {
        if (count*64 <= (int) ids.size()) {
            std::partial_sort(ids.begin(), ids.begin() + count, ids.end(), compare);
        } else {
            std::nth_element(ids.begin(), ids.begin() + count - 1, ids.end(), compare);
        }
    }
    ids.resize(count);
    return ids;
}

void dsv41_index(ggml_tensor * dst, int ith, int nth, void * userdata) {
    GGML_UNUSED(ith);
    GGML_UNUSED(nth);
    const auto & op = *(const dsv41_op_data *) userdata;
    const auto & hp = op.memory->hparams;
    const auto * q = dst->src[0];
    const auto * weights = dst->src[1];
    const auto * pos = (const int32_t *) dst->src[2]->data;
    const int ratio = hp.dsv4_compress_ratios[op.il];
    const bool candidate_source = op.il == hp.dsv41_candidate_source_layer;
    const bool uses_candidates = hp.dsv41_candidate_source_layer >= 0 && op.il > hp.dsv41_candidate_source_layer;
    const int block_size = hp.dsv41_candidate_block_size;
    std::vector<float> key(hp.indexer_head_size);
    const int width = (pos[dst->ne[1] - 1] + 1)/ratio;
    for (int64_t it = 0; it < dst->ne[1]; ++it) {
        const int visible = (pos[it] + 1)/ratio;
        auto * out = (int32_t *) ((char *) dst->data + it*dst->nb[1]);
        std::fill_n(out, dst->ne[0], -1);
        std::vector<float> scores(width, -INFINITY);
        for (int ik = 0; ik < visible; ++ik) {
            if (uses_candidates) {
                const auto * blocks = (const int32_t *) ((const char *) dst->src[4]->data + it*dst->src[4]->nb[1]) + hp.indexer_top_k;
                if (std::find(blocks, blocks + hp.dsv41_candidate_topk_blocks, ik/block_size) == blocks + hp.dsv41_candidate_topk_blocks) {
                    continue;
                }
            }
            op.memory->read_index(op.il, ik, key.data());
            float sum = 0;
            for (uint32_t ih = 0; ih < hp.indexer_n_head; ++ih) {
                const auto * query = (const float *) ((const char *) q->data + ih*q->nb[1] + it*q->nb[2]);
                float dot = 0;
                for (uint32_t j = 0; j < hp.indexer_head_size; ++j) {
                    dot += query[j]*key[j];
                }
                sum += dsv41_bf16(std::max(dsv41_bf16(dot), 0.0f)*dsv41_frow(weights, it)[ih]);
            }
            scores[ik] = dsv41_bf16(sum);
        }
        if (candidate_source && visible > 0) {
            std::vector<float> blocks((width + block_size - 1)/block_size, -INFINITY);
            for (int ik = 0; ik < visible; ++ik) {
                blocks[ik/block_size] = std::max(blocks[ik/block_size], scores[ik]);
            }
            blocks[(visible - 1)/block_size] = INFINITY;
            const auto selected = dsv41_topk(blocks, hp.dsv41_candidate_topk_blocks);
            int count = 0;
            for (auto id : selected) {
                if (blocks[id] > -INFINITY) { out[hp.indexer_top_k + count++] = id; }
            }
        }
        auto selected = dsv41_topk(scores, hp.indexer_top_k);
        selected.erase(std::remove_if(selected.begin(), selected.end(), [&](int32_t id) { return scores[id] == -INFINITY; }), selected.end());
        std::sort(selected.begin(), selected.end());
        std::copy(selected.begin(), selected.end(), out);
    }
}

void dsv41_attention(ggml_tensor * dst, int ith, int nth, void * userdata) {
    GGML_UNUSED(ith);
    GGML_UNUSED(nth);
    const auto & op = *(const dsv41_op_data *) userdata;
    const auto & hp = op.memory->hparams;
    const auto * q = dst->src[0];
    const auto * raw = dst->src[1];
    const auto * sinks = (const float *) dst->src[2]->data;
    const auto * pos = (const int32_t *) dst->src[3]->data;
    const auto * index = dst->src[4];
    const int dim = dst->ne[0], heads = dst->ne[1], nt = dst->ne[2];
    const int window = pos[0] == 0 ? std::min<int>(nt, hp.n_swa) : hp.n_swa;
    const int topk = index ? std::min<int>(hp.indexer_top_k, (pos[nt - 1] + 1)/hp.dsv4_compress_ratios[op.il]) : 0;
    std::vector<float> kv((window + topk)*dim), accumulator(dim);
    std::vector<bool> valid(window + topk);
    for (int it = 0; it < nt; ++it) {
        op.memory->write_raw(op.il, pos[it], dsv41_frow(raw, it));
    }
    const float scale = 1.0f/std::sqrt(float(dim));
    for (int it = 0; it < nt; ++it) {
        const int first = pos[it] - int(hp.n_swa) + 1;
        for (int slot = 0; slot < window + topk; ++slot) {
            auto * row = kv.data() + slot*dim;
            if (slot < window) {
                const int p = (pos[0] == 0 ? std::max(0, first) : first) + slot;
                valid[slot] = p >= 0 && p <= pos[it];
                if (valid[slot]) { op.memory->read_raw(op.il, p, row); }
            } else {
                const auto id = ((const int32_t *) ((const char *) index->data + it*index->nb[1]))[slot - window];
                valid[slot] = id >= 0;
                if (valid[slot]) { op.memory->read_kv(op.il, id, row); }
            }
            if (!valid[slot]) { std::fill_n(row, dim, 0.0f); }
        }
        for (int ih = 0; ih < heads; ++ih) {
            const auto * query = (const float *) ((const char *) q->data + ih*q->nb[1] + it*q->nb[2]);
            std::fill(accumulator.begin(), accumulator.end(), 0.0f);
            float maximum = -1e30f, denominator = 0;
            for (int start = 0; start < window + topk; start += 64) {
                const int count = std::min(64, window + topk - start);
                float scores[64], weights[64], next_maximum = maximum;
                for (int slot = 0; slot < count; ++slot) {
                    float dot = 0;
                    for (int j = 0; j < dim; ++j) { dot += query[j]*kv[(start + slot)*dim + j]; }
                    scores[slot] = valid[start + slot] ? dot*scale : -INFINITY;
                    next_maximum = std::max(next_maximum, scores[slot]);
                }
                const float correction = std::exp(maximum - next_maximum);
                float sum = 0;
                for (int slot = 0; slot < count; ++slot) {
                    const float probability = std::exp(scores[slot] - next_maximum);
                    sum += probability;
                    weights[slot] = dsv41_bf16(probability);
                }
                denominator = denominator*correction + sum;
                for (int j = 0; j < dim; ++j) {
                    float value = 0;
                    for (int slot = 0; slot < count; ++slot) { value += weights[slot]*kv[(start + slot)*dim + j]; }
                    accumulator[j] = accumulator[j]*correction + value;
                }
                maximum = next_maximum;
            }
            denominator += std::exp(sinks[ih] - maximum);
            auto * out = (float *) ((char *) dst->data + ih*dst->nb[1] + it*dst->nb[2]);
            for (int j = 0; j < dim; ++j) { out[j] = dsv41_bf16(accumulator[j]/denominator); }
        }
    }
}

struct dsv41_graph : public llm_graph_context {
    const llama_model_deepseek41 & model;
    llm_graph_input_dsv41 * input;
    ggml_tensor * positions;
    std::vector<ggml_tensor *> published;
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
        auto * flat = ggml_reshape_2d(ctx0, x, hc*n_embd, n_tokens);
        auto * mixes = ggml_mul(ctx0, mm_f32(fn, flat), rstd(flat));
        cb(mixes, ffn ? "dsv41_ffn_mixes" : "dsv41_attn_mixes", il);
        auto * split = ggml_dsv41_hc_split(ctx0, mixes, scale, base, hparams.dsv4_hc_eps, hparams.dsv4_hc_sinkhorn_iters);
        hc_mixes result = {
            ggml_view_2d(ctx0, split, hc, n_tokens, split->nb[1], 0),
            ggml_view_2d(ctx0, split, hc, n_tokens, split->nb[1], hc*sizeof(float)),
            ggml_view_3d(ctx0, split, hc, hc, n_tokens, hc*sizeof(float), split->nb[1], 2*hc*sizeof(float)),
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
            rotations = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_rot, n_tokens);
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
                auto * row = ggml_view_2d(ctx0, residual, n_embd, n_tokens, residual->nb[2], src*residual->nb[1]);
                auto * weight = ggml_view_2d(ctx0, mix.comb, 1, n_tokens, mix.comb->nb[2], dst*mix.comb->nb[0] + src*mix.comb->nb[1]);
                auto * term = ggml_mul(ctx0, row, weight);
                sum = sum ? ggml_add(ctx0, sum, term) : term;
            }
            auto * post = ggml_view_2d(ctx0, mix.post, 1, n_tokens, mix.post->nb[1], dst*mix.post->nb[0]);
            auto * copy = ggml_reshape_3d(ctx0, ggml_add(ctx0, ggml_mul(ctx0, x, post), sum), n_embd, 1, n_tokens);
            out = out ? ggml_concat(ctx0, out, copy, 1) : copy;
        }
        return bf16(out);
    }

    ggml_tensor * custom(ggml_type type, int64_t n0, int64_t n1, int64_t n2,
            std::initializer_list<ggml_tensor *> args, ggml_custom_op_t fn, int il) {
        std::vector<ggml_tensor *> sources(args);
        return ggml_custom_4d(ctx0, type, n0, n1, n2, 1, sources.data(), sources.size(), fn, 1, input->add_op(il));
    }

    ggml_tensor * engram(ggml_tensor * x, size_t ie) const {
        const int il = model.engram.layers[ie];
        const auto & layer = model.layers[il];
        const int64_t hc = hparams.dsv4_hc_mult;
        auto * embedding = bf16(input->engram[ie]);
        cb(embedding, "dsv41_engram_embedding", il);
        auto * kv = linear(layer.engram_kv, embedding, hparams.dsv41_dense_act_fp8);
        cb(kv, "dsv41_engram_kv", il);
        auto * key = ggml_view_3d(ctx0, kv, n_embd, hc, n_tokens, n_embd*sizeof(float), kv->nb[1], 0);
        auto * value = ggml_view_2d(ctx0, kv, n_embd, n_tokens, kv->nb[1], hc*n_embd*sizeof(float));
        auto * weight = ggml_mul(ctx0, ggml_cast(ctx0, layer.engram_q_weight, GGML_TYPE_F32), ggml_cast(ctx0, layer.engram_k_weight, GGML_TYPE_F32));
        auto * out = bf16(ggml_dsv41_engram(ctx0, x, key, value, weight, norm_rms_eps, 1e-6f));
        cb(out, "dsv41_engram", il);
        return out;
    }

    ggml_tensor * attention(ggml_tensor * x, int il) {
        const auto & layer = model.layers[il];
        const bool fp8 = hparams.dsv41_dense_act_fp8;
        const int64_t dim = hparams.n_embd_head_k(il), heads = hparams.n_head(il);
        const int ratio = hparams.dsv4_compress_ratios[il];
        auto * qa = linear(layer.wq_a, x, fp8);
        cb(qa, "dsv41_q_a", il);
        auto * qr = norm(qa, layer.attn_q_a_norm);
        cb(qr, "dsv41_q_norm", il);
        auto * q = linear(layer.wq_b, qr, fp8);
        cb(q, "dsv41_q_b", il);
        q = rope(ggml_reshape_3d(ctx0, q, dim, heads, n_tokens), il);
        cb(q, "dsv41_q", il);
        auto * raw = norm(linear(layer.wkv, x, fp8), layer.attn_kv_norm);
        raw = rope(ggml_reshape_3d(ctx0, raw, dim, 1, n_tokens), il);
        raw = ggml_reshape_2d(ctx0, raw, dim, n_tokens);
        cb(raw, "dsv41_raw_kv", il);
        if (hparams.dsv41_kv_source[il] == il) {
            ggml_tensor * latent;
            if (ratio == 2) {
                auto * kv = mm_f32(layer.attn_comp_wkv, x);
                auto * score = mm_f32(layer.attn_comp_wgate, x);
                latent = custom(GGML_TYPE_F32, dim, n_tokens, 1, {kv, score, positions}, dsv41_pool, il);
            } else {
                latent = linear(layer.attn_comp_wkv, x, false);
            }
            latent = norm(latent, layer.attn_comp_norm);
            cb(latent, "dsv41_latent", il);
            auto * key = norm(linear(layer.indexer_attn_k, latent, false), layer.indexer_k_norm);
            key = rope(ggml_reshape_3d(ctx0, key, hparams.indexer_head_size, 1, n_tokens), il, false, ratio == 2);
            key = ggml_reshape_2d(ctx0, key, hparams.indexer_head_size, n_tokens);
            cb(key, "dsv41_index_key", il);
            auto * kv = rope(ggml_reshape_3d(ctx0, latent, dim, 1, n_tokens), il, false, ratio == 2);
            kv = ggml_reshape_2d(ctx0, kv, dim, n_tokens);
            published[il] = custom(GGML_TYPE_F32, 1, 1, 1, {kv, key, positions}, dsv41_publish, il);
        }
        ggml_tensor * selected = nullptr;
        if (ratio) {
            if (hparams.dsv41_index_source[il] == il) {
                auto * qi = linear(layer.indexer_attn_q_b, qr, fp8);
                qi = rope(ggml_reshape_3d(ctx0, qi, hparams.indexer_head_size, hparams.indexer_n_head, n_tokens), il);
                qi = ggml_dsv41_act_quant(ctx0, qi, GGML_DSV41_QUANT_MXFP4);
                cb(qi, "dsv41_index_query", il);
                auto * weights = linear(layer.indexer_proj, x, false);
                weights = bf16(ggml_scale(ctx0, weights, 1.0f/std::sqrt(float(hparams.indexer_head_size*hparams.indexer_n_head))));
                cb(weights, "dsv41_index_weights", il);
                auto * pub = published[hparams.dsv41_kv_source[il]];
                indices[il] = custom(GGML_TYPE_I32, hparams.indexer_top_k + hparams.dsv41_candidate_topk_blocks, n_tokens, 1,
                        {qi, weights, positions, pub, candidates}, dsv41_index, il);
                cb(indices[il], "dsv41_index_selected", il);
                if (il == hparams.dsv41_candidate_source_layer) { candidates = indices[il]; }
            }
            selected = indices[hparams.dsv41_index_source[il]];
        }
        auto * out = custom(GGML_TYPE_F32, dim, heads, n_tokens, {q, raw, layer.attn_sinks, positions, selected}, dsv41_attention, il);
        cb(out, "dsv41_sparse_attention", il);
        out = rope(out, il, true);
        cb(out, "dsv41_derope", il);
        const int64_t groups = hparams.dsv4_o_group_count, rank = hparams.dsv4_o_lora_rank;
        out = ggml_permute(ctx0, ggml_reshape_3d(ctx0, out, heads*dim/groups, groups, n_tokens), 0, 2, 1, 3);
        auto * wa = ggml_reshape_3d(ctx0, layer.wo_a, heads*dim/groups, rank, groups);
        auto * oa = bf16(mm_f32(wa, out));
        oa = ggml_cont_2d(ctx0, ggml_permute(ctx0, oa, 0, 2, 1, 3), rank*groups, n_tokens);
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
        auto * weights = ggml_get_rows(ctx0, ggml_reshape_3d(ctx0, scores, 1, n_expert, n_tokens), ids);
        weights = ggml_reshape_2d(ctx0, weights, used, n_tokens);
        if (used > 1) {
            weights = ggml_div(ctx0, weights, ggml_scale_bias(ctx0, ggml_sum_rows(ctx0, weights), 1.0f, 1e-20f));
        }
        weights = ggml_scale(ctx0, weights, hparams.expert_weights_scale);
        cb(weights, "dsv41_expert_weights", il);
        weights = ggml_reshape_3d(ctx0, weights, 1, used, n_tokens);
        auto * cur = hparams.dsv41_expert_act_fp8 ? ggml_dsv41_act_quant(ctx0, x, GGML_DSV41_QUANT_MXFP8) : x;
        cur = ggml_reshape_3d(ctx0, cur, n_embd, 1, n_tokens);
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
            auto * expert = ggml_view_2d(ctx0, experts, n_embd, n_tokens, experts->nb[2], i*experts->nb[1]);
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
        llm_graph_context(params), model(model), published(n_layer), indices(n_layer) {
        const auto * memory_context = dynamic_cast<const llama_memory_dsv41_context *>(mctx);
        dsv41_require(memory_context != nullptr, "missing compact state context");
        auto owner = std::make_unique<llm_graph_input_dsv41>(model, memory_context->memory);
        input = owner.get();
        input->n_tokens = n_tokens;
        input->init_rope(n_rot, freq_base, hparams.dsv4_compress_rope_base, freq_scale, ext_factor, n_ctx_orig, beta_fast, beta_slow);
        const auto & e = model.engram;
        for (size_t ie = 0; ie < e.layers.size(); ++ie) {
            auto * rows = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, (e.ngram_size - 1)*e.n_heads*e.head_dim, n_tokens);
            ggml_set_input(rows);
            input->engram.push_back(rows);
        }
        res->add_input(std::move(owner));
        positions = build_inp_pos();
        auto * out_ids = build_inp_out_ids();
        const int64_t hc = hparams.dsv4_hc_mult;
        auto * x = build_inp_embd(model.tok_embd);
        x = ggml_repeat_4d(ctx0, ggml_reshape_3d(ctx0, x, n_embd, 1, n_tokens), n_embd, hc, n_tokens, 1);
        ggml_tensor * pre = nullptr;
        size_t ie = 0;
        for (int il = 0; il < n_layer; ++il) {
            const auto & layer = model.layers[il];
            if (ie < e.layers.size() && e.layers[ie] == (uint32_t) il) { x = engram(x, ie++); }
            const auto attn_mix = mix(x, layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base, il, false);
            auto * collapsed = pre ? bf16(ggml_dsv4_hc_pre(ctx0, x, pre)) : ggml_cont(ctx0, ggml_view_2d(ctx0, x, n_embd, n_tokens, x->nb[2], 0));
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
