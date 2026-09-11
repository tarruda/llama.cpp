#include "models.h"

#include <algorithm>
#include <stdexcept>

static void dsv41_require(bool valid, const char * message) {
    if (!valid) {
        throw std::runtime_error(std::string("DeepSeek-V4.1: ") + message);
    }
}

void llama_model_deepseek41::load_arch_hparams(llama_model_loader & ml) {
    dsv41_require(hparams.n_layer_nextn == 0, "draft layers require a separate model");
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
        layer.ffn_exp_probs_b_vl = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B_VL, "bias", il), {n_expert}, 0);
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

std::unique_ptr<llm_graph_context> llama_model_deepseek41::build_arch_graph(const llm_graph_params & params) const {
    GGML_UNUSED(params);
    throw std::runtime_error("DeepSeek-V4.1 graph support is not implemented yet");
}
