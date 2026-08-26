#include "models.h"
#include "llama-kv-cache-msa.h"
#include "llama-memory-qwen4.h"
#include "llama-memory-recurrent.h"

#include <algorithm>
#include <cinttypes>
#include <limits>
#include <numeric>
#include <stdexcept>

static ggml_tensor * qwen4_hc_mean(ggml_context * ctx, ggml_tensor * input) {
    const int64_t n_hc = input->ne[1];
    ggml_tensor * result = ggml_view_2d(ctx, input, input->ne[0], input->ne[2], input->nb[2], 0);
    for (int64_t ih = 1; ih < n_hc; ++ih) {
        result = ggml_add(ctx, result, ggml_view_2d(ctx, input, input->ne[0], input->ne[2], input->nb[2], ih * input->nb[1]));
    }
    return ggml_scale(ctx, result, 1.0f / n_hc);
}

static ggml_tensor * qwen4_l2_norm(ggml_context * ctx, ggml_tensor * input) {
    ggml_tensor * norm = ggml_sum_rows(ctx, ggml_sqr(ctx, input));
    norm = ggml_sqrt(ctx, ggml_scale_bias(ctx, norm, 1.0f, 1e-6f));
    return ggml_div(ctx, input, norm);
}

class llm_graph_input_qwen4_qsa : public llm_graph_input_i {
public:
    llm_graph_input_qwen4_qsa(
            const llama_memory_qwen4_context * mctx,
            ggml_tensor * kq_mask,
            uint32_t ratio,
            uint32_t block_topk) :
        mctx(mctx),
        kq_mask(kq_mask),
        ratio(ratio),
        block_topk(block_topk) {
    }

    void set_input(const llama_ubatch * ubatch) override {
        mctx->set_input_qsa_layout(block_cells, block_pos, block_mask, selected, kq_mask, ubatch, ratio, block_topk);
    }

    bool can_reuse(const llm_graph_params & params) override {
        const auto * mctx = static_cast<const llama_memory_qwen4_context *>(params.mctx);
        this->mctx = mctx;

        const int64_t n_kv = mctx->get_attn()->get_base()->get_n_kv();
        const int64_t n_blocks = n_kv / ratio;

        bool result = true;
        result &= n_kv > (int64_t) block_topk * ratio + ratio - 1;
        result &= block_cells->ne[1] == n_blocks;
        result &= block_cells->ne[2] == params.ubatch.n_tokens;
        result &= block_pos->ne[2] == params.ubatch.n_tokens;
        result &= block_mask->ne[1] == params.ubatch.n_tokens;
        result &= selected->ne[0] == n_kv;
        result &= selected->ne[1] == params.ubatch.n_tokens;
        return result;
    }

    ggml_tensor * block_cells = nullptr;
    ggml_tensor * block_pos = nullptr;
    ggml_tensor * block_mask = nullptr;
    ggml_tensor * selected = nullptr;

    const llama_memory_qwen4_context * mctx;
    ggml_tensor * kq_mask;
    const uint32_t ratio;
    const uint32_t block_topk;
};

void llama_model_qwen4exp::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,        hparams.n_ff_exp);
    ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,       hparams.f_norm_rms_eps);
    ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS,    hparams.rope_sections, 4, true);

    ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
    ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
    ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
    ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
    ml.get_key(LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);

    ml.get_key(LLM_KV_HYPER_CONNECTION_COUNT,    hparams.qwen4_hc_count);
    ml.get_key(LLM_KV_HYPER_CONNECTION_LOW_RANK, hparams.qwen4_hc_low_rank);
    if (hparams.qwen4_hc_count <= 1 || hparams.qwen4_hc_low_rank == 0) {
        throw std::runtime_error(format("invalid Qwen4 hyper-connection dimensions: count %u, low rank %u", hparams.qwen4_hc_count, hparams.qwen4_hc_low_rank));
    }
    hparams.n_embd_nextn_impl = hparams.qwen4_hc_count * hparams.n_embd;

    ml.get_key(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, hparams.indexer_n_head);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, hparams.indexer_head_size);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_TOP_K,      hparams.indexer_top_k);
    if (hparams.indexer_n_head == 0 || hparams.indexer_head_size == 0 || hparams.indexer_top_k == 0) {
        throw std::runtime_error("invalid Qwen4 sparse-attention dimensions");
    }

    ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS, hparams.n_layer_nextn, false);
    if (hparams.n_layer_nextn >= hparams.n_layer_all) {
        throw std::runtime_error(format("invalid Qwen4 MTP layer count: %u of %u layers", hparams.n_layer_nextn, hparams.n_layer_all));
    }

    if (!ml.get_key_or_arr(LLM_KV_ATTENTION_RECURRENT_LAYERS, hparams.is_recr_impl, hparams.n_layer_all, false)) {
        uint32_t full_attn_interval = 4;
        ml.get_key(LLM_KV_FULL_ATTENTION_INTERVAL, full_attn_interval, false);
        if (full_attn_interval == 0) {
            throw std::runtime_error("invalid Qwen4 full-attention interval: 0");
        }
        for (uint32_t il = 0; il < hparams.n_layer_all; ++il) {
            hparams.is_recr_impl[il] = il < hparams.n_layer() && (il + 1) % full_attn_interval != 0;
        }
    }
    ml.get_key_or_arr(LLM_KV_ATTENTION_COMPRESS_RATIOS, hparams.qwen4_compress_ratios, hparams.n_layer_all, true);
    uint32_t qsa_ratio = 0;
    for (uint32_t il = 0; il < hparams.n_layer_all; ++il) {
        const uint32_t ratio = hparams.qwen4_compress_ratios[il];
        if (hparams.is_recr(il)) {
            if (ratio != 0) {
                throw std::runtime_error(format("Qwen4 recurrent layer %u has a sparse-attention compression ratio", il));
            }
        } else if (ratio == 0 || hparams.indexer_top_k % ratio != 0) {
            throw std::runtime_error(format("invalid Qwen4 sparse-attention compression ratio %u at layer %u", ratio, il));
        } else if (qsa_ratio != 0 && qsa_ratio != ratio) {
            throw std::runtime_error("Qwen4 sparse-attention compression ratios must match across layers");
        } else {
            qsa_ratio = ratio;
        }
    }

    std::vector<int32_t> ple_layers;
    std::vector<int64_t> ple_multipliers;
    std::vector<int64_t> ple_offsets;
    std::vector<int64_t> ple_vocab_sizes;
    ml.get_key(LLM_KV_EMBEDDING_LENGTH_PER_LAYER, hparams.n_embd_per_layer, false);
    if (!ml.get_arr(LLM_KV_PER_LAYER_EMBEDDING_LAYERS, ple_layers, false)) {
        ml.get_arr("qwen4exp.ple.layers", ple_layers, false);
    }
    if (!ple_layers.empty()) {
        if (!ml.get_key(LLM_KV_PER_LAYER_EMBEDDING_NGRAM_SIZE, hparams.qwen4_ple_ngram, false)) {
            ml.get_key("qwen4exp.ple.ngram_size", hparams.qwen4_ple_ngram);
        }
        if (!ml.get_key(LLM_KV_PER_LAYER_EMBEDDING_HEADS_PER_NGRAM, hparams.qwen4_ple_heads, false)) {
            ml.get_key("qwen4exp.ple.heads_per_ngram", hparams.qwen4_ple_heads);
        }
        ml.get_key(LLM_KV_PER_LAYER_EMBEDDING_VOCAB_SIZE_DIVISOR, hparams.qwen4_ple_vocab_div, false);
        if (!ml.get_key(LLM_KV_PER_LAYER_EMBEDDING_CONV_KERNEL, hparams.qwen4_ple_conv, false)) {
            ml.get_key("qwen4exp.ple.conv_kernel", hparams.qwen4_ple_conv);
        }
        if (!ml.get_key(LLM_KV_PER_LAYER_EMBEDDING_EOS_TOKEN_ID, hparams.qwen4_ple_eos, false)) {
            ml.get_key("qwen4exp.ple.eos_token_id", hparams.qwen4_ple_eos);
        }
        if (!ml.get_arr(LLM_KV_PER_LAYER_EMBEDDING_LAYER_MULTIPLIERS, ple_multipliers, false)) {
            ml.get_arr("qwen4exp.ple.layer_multipliers", ple_multipliers);
        }
        if (!ml.get_arr(LLM_KV_PER_LAYER_EMBEDDING_HEAD_OFFSETS, ple_offsets, false)) {
            ml.get_arr("qwen4exp.ple.head_offsets", ple_offsets);
        }
        if (!ml.get_arr(LLM_KV_PER_LAYER_EMBEDDING_HEAD_VOCAB_SIZES, ple_vocab_sizes, false)) {
            ml.get_arr("qwen4exp.ple.head_vocab_sizes", ple_vocab_sizes);
        }

        if (ple_layers.size() != 1 || hparams.qwen4_ple_ngram < 2 || hparams.qwen4_ple_heads == 0 || hparams.qwen4_ple_vocab_div == 0 || hparams.qwen4_ple_conv == 0 || hparams.n_embd_per_layer == 0) {
            throw std::runtime_error("invalid Qwen4 per-layer embedding dimensions");
        }
        const uint64_t n_ple_heads = uint64_t(hparams.qwen4_ple_ngram - 1) * hparams.qwen4_ple_heads;
        if (n_ple_heads > LLAMA_MAX_LAYERS || ple_multipliers.size() != hparams.qwen4_ple_ngram || ple_offsets.size() != n_ple_heads || ple_vocab_sizes.size() != n_ple_heads) {
            throw std::runtime_error("invalid Qwen4 per-layer embedding metadata lengths");
        }
        const int32_t ple_layer = ple_layers[0];
        if (ple_layer < 0 || uint32_t(ple_layer) >= hparams.n_layer() || !hparams.is_recr(ple_layer)) {
            throw std::runtime_error(format("invalid Qwen4 per-layer embedding layer: %d", ple_layer));
        }
        int64_t offset = 0;
        for (size_t ih = 0; ih < n_ple_heads; ++ih) {
            if (ple_vocab_sizes[ih] <= 0 || ple_offsets[ih] != offset) {
                throw std::runtime_error("invalid Qwen4 per-layer embedding table partition");
            }
            if (ple_vocab_sizes[ih] > std::numeric_limits<int64_t>::max() - offset) {
                throw std::runtime_error("Qwen4 per-layer embedding table is too large");
            }
            offset += ple_vocab_sizes[ih];
        }
        hparams.qwen4_ple_layer = ple_layer;
        std::copy(ple_multipliers.begin(), ple_multipliers.end(), hparams.qwen4_ple_multipliers.begin());
        std::copy(ple_offsets.begin(), ple_offsets.end(), hparams.qwen4_ple_offsets.begin());
        std::copy(ple_vocab_sizes.begin(), ple_vocab_sizes.end(), hparams.qwen4_ple_vocab_sizes.begin());
    }

    hparams.expert_weights_norm = true;
    type = LLM_TYPE_UNKNOWN;
}

void llama_model_qwen4exp::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    if (n_expert == 0 || n_expert_used == 0) {
        throw std::runtime_error("Qwen4 model requires routed experts");
    }

    const bool mtp_only = hparams.n_layer_nextn > 0 && ml.get_weight("blk.0.hc_attn_norm.weight") == nullptr;
    const int trunk_flags = mtp_only ? TENSOR_NOT_REQUIRED : 0;
    const int mtp_flags = !ml.load_mtp ? TENSOR_SKIP : 0;
    const int64_t n_hc = hparams.qwen4_hc_count;
    const int64_t n_hc_embd = n_hc * n_embd;
    const int64_t n_hc_rank = hparams.qwen4_hc_low_rank;
    const int64_t n_ff_exp = hparams.n_ff_exp ? hparams.n_ff_exp : n_ff / n_expert_used;
    const int64_t n_ff_shexp = hparams.n_ff_shexp ? hparams.n_ff_shexp : n_ff;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);
    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED);
    if (output == nullptr) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, TENSOR_DUPLICATED);
    }

    hc_head_norm = create_tensor(tn(LLM_TENSOR_HC_HEAD_NORM, "weight"), { n_hc_embd }, trunk_flags);
    hc_head_down = create_tensor(tn(LLM_TENSOR_HC_HEAD_DOWN, "weight"), { n_hc_embd, n_hc_rank }, trunk_flags);
    hc_head_up   = create_tensor(tn(LLM_TENSOR_HC_HEAD_UP,   "weight"), { n_hc_rank, n_hc_embd }, trunk_flags);

    if (hparams.qwen4_ple_layer >= 0) {
        const int64_t n_ple_heads = (hparams.qwen4_ple_ngram - 1) * hparams.qwen4_ple_heads;
        const int64_t n_rows_valid = std::accumulate(hparams.qwen4_ple_vocab_sizes.begin(), hparams.qwen4_ple_vocab_sizes.begin() + n_ple_heads, int64_t(0));
        const int64_t remainder = n_rows_valid % hparams.qwen4_ple_vocab_div;
        const int64_t padding = remainder == 0 ? 0 : hparams.qwen4_ple_vocab_div - remainder;
        if (n_rows_valid > std::numeric_limits<int64_t>::max() - padding) {
            throw std::runtime_error("Qwen4 per-layer embedding table is too large");
        }
        int64_t n_rows = n_rows_valid + padding;
        const auto tensor_name = tn(LLM_TENSOR_PER_LAYER_TOKEN_EMBD, "weight");
        const ggml_tensor * tensor = ml.get_tensor_meta(tensor_name.str().c_str());
        if (tensor != nullptr) {
            if (tensor->ne[1] != n_rows_valid && tensor->ne[1] != n_rows) {
                throw std::runtime_error(format("invalid Qwen4 per-layer embedding row count: expected %" PRId64 " or %" PRId64 ", got %" PRId64, n_rows_valid, n_rows, tensor->ne[1]));
            }
            n_rows = tensor->ne[1];
        }
        per_layer_tok_embd = create_tensor(tensor_name, { hparams.n_embd_per_layer, n_rows }, trunk_flags);
    }

    const int64_t head_k_dim = hparams.ssm_d_state;
    const int64_t n_k_heads = hparams.ssm_n_group;
    const int64_t n_v_heads = hparams.ssm_dt_rank;
    const int64_t key_dim = head_k_dim * n_k_heads;
    const int64_t value_dim = hparams.ssm_d_inner;
    const int64_t conv_dim = 2 * key_dim + value_dim;

    auto load_hc = [&](llama_layer & layer, int il, int flags) {
        layer.hc_attn_norm   = create_tensor(tn(LLM_TENSOR_HC_ATTN_NORM,   "weight", il), { n_hc_embd }, flags);
        layer.hc_attn_down   = create_tensor(tn(LLM_TENSOR_HC_ATTN_DOWN,   "weight", il), { n_hc_embd, n_hc_rank }, flags);
        layer.hc_attn_up     = create_tensor(tn(LLM_TENSOR_HC_ATTN_UP,     "weight", il), { n_hc_rank, n_hc_embd }, flags);
        layer.hc_attn_inject = create_tensor(tn(LLM_TENSOR_HC_ATTN_INJECT, "weight", il), { n_hc_embd, n_hc }, flags);
        layer.hc_ffn_norm    = create_tensor(tn(LLM_TENSOR_HC_FFN_NORM,    "weight", il), { n_hc_embd }, flags);
        layer.hc_ffn_down    = create_tensor(tn(LLM_TENSOR_HC_FFN_DOWN,    "weight", il), { n_hc_embd, n_hc_rank }, flags);
        layer.hc_ffn_up      = create_tensor(tn(LLM_TENSOR_HC_FFN_UP,      "weight", il), { n_hc_rank, n_hc_embd }, flags);
        layer.hc_ffn_inject  = create_tensor(tn(LLM_TENSOR_HC_FFN_INJECT,  "weight", il), { n_hc_embd, n_hc }, flags);
    };

    auto load_moe = [&](llama_layer & layer, int il, int flags) {
        layer.ffn_gate_inp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", il), { n_embd, n_expert }, flags);
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", il), { n_ff_exp, n_embd, n_expert }, flags);
        create_tensor_gate_up_exps(layer, il, n_embd, n_ff_exp, n_expert, flags);
        layer.ffn_gate_inp_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP_SHEXP, "weight", il), { n_embd }, flags);
        layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", il), { n_embd, n_ff_shexp }, flags);
        layer.ffn_up_shexp = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP, "weight", il), { n_embd, n_ff_shexp }, flags);
        layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", il), { n_ff_shexp, n_embd }, flags);
    };

    auto load_attention = [&](llama_layer & layer, int il, int flags) {
        create_tensor_qkv(layer, il, n_embd, n_embd_head_k * n_head * 2, n_embd_k_gqa, n_embd_v_gqa, flags);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", il), { n_embd_head_k * n_head, n_embd }, flags);
        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", il), { n_embd_head_k }, flags);
        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", il), { n_embd_head_k }, flags);
        layer.index_q_proj = create_tensor(tn(LLM_TENSOR_INDEXER_Q_PROJ, "weight", il), { n_embd, hparams.indexer_n_head * hparams.indexer_head_size }, flags);
        layer.index_k_proj = create_tensor(tn(LLM_TENSOR_INDEXER_K_PROJ, "weight", il), { n_embd, hparams.indexer_head_size }, flags);
        layer.index_q_norm = create_tensor(tn(LLM_TENSOR_INDEXER_Q_NORM, "weight", il), { hparams.indexer_head_size }, flags);
        layer.index_k_norm = create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM, "weight", il), { hparams.indexer_head_size }, flags);
    };

    for (int il = 0; il < n_layer; ++il) {
        auto & layer = layers[il];
        load_hc(layer, il, trunk_flags);
        load_moe(layer, il, trunk_flags);

        if (hparams.is_recr(il)) {
            layer.wqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "weight", il), { n_embd, conv_dim }, trunk_flags | TENSOR_NOT_REQUIRED);
            layer.wqkv_gate = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "weight", il), { n_embd, value_dim }, trunk_flags | TENSOR_NOT_REQUIRED);
            layer.ssm_conv1d = create_tensor(tn(LLM_TENSOR_SSM_CONV1D, "weight", il), { hparams.ssm_d_conv, conv_dim }, trunk_flags);
            layer.ssm_dt = create_tensor(tn(LLM_TENSOR_SSM_DT, "bias", il), { n_v_heads }, trunk_flags);
            layer.ssm_a = create_tensor(tn(LLM_TENSOR_SSM_A_NOSCAN, nullptr, il), { n_v_heads }, trunk_flags);
            layer.ssm_beta = create_tensor(tn(LLM_TENSOR_SSM_BETA, "weight", il), { n_embd, n_v_heads }, trunk_flags);
            layer.ssm_alpha = create_tensor(tn(LLM_TENSOR_SSM_ALPHA, "weight", il), { n_embd, n_v_heads }, trunk_flags);
            layer.ssm_norm = create_tensor(tn(LLM_TENSOR_SSM_NORM, "weight", il), { hparams.ssm_d_state }, trunk_flags);
            layer.ssm_out = create_tensor(tn(LLM_TENSOR_SSM_OUT, "weight", il), { value_dim, n_embd }, trunk_flags);
        } else {
            load_attention(layer, il, trunk_flags);
        }

        if (il == hparams.qwen4_ple_layer) {
            const int64_t n_ple_heads = (hparams.qwen4_ple_ngram - 1) * hparams.qwen4_ple_heads;
            const int64_t n_ple_embd = hparams.n_embd_per_layer * n_ple_heads;
            layer.ple_key = create_tensor(tn(LLM_TENSOR_PLE_KEY, "weight", il), { n_ple_embd, n_hc_embd }, trunk_flags);
            layer.ple_value = create_tensor(tn(LLM_TENSOR_PLE_VALUE, "weight", il), { n_ple_embd, n_embd }, trunk_flags);
            layer.ple_norm_key = create_tensor(tn(LLM_TENSOR_PLE_NORM_KEY, "weight", il), { n_hc_embd }, trunk_flags);
            layer.ple_norm_query = create_tensor(tn(LLM_TENSOR_PLE_NORM_QUERY, "weight", il), { n_hc_embd }, trunk_flags);
            layer.ple_norm_conv = create_tensor(tn(LLM_TENSOR_PLE_NORM_CONV, "weight", il), { n_hc_embd }, trunk_flags);
            layer.ple_conv1d = create_tensor(tn(LLM_TENSOR_PLE_CONV1D, "weight", il), { hparams.qwen4_ple_conv, n_hc_embd }, trunk_flags);
        }
    }

    for (int il = n_layer; il < n_layer_all; ++il) {
        auto & layer = layers[il];
        load_hc(layer, il, mtp_flags);
        load_moe(layer, il, mtp_flags);
        load_attention(layer, il, mtp_flags);
        layer.nextn.enorm = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM, "weight", il), { n_embd }, mtp_flags);
        layer.nextn.hnorm = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM, "weight", il), { n_hc_embd }, mtp_flags);
        layer.nextn.embed_proj = create_tensor(tn(LLM_TENSOR_NEXTN_EMBED_PROJ, "weight", il), { n_embd, n_embd }, mtp_flags);
        layer.nextn.hidden_proj = create_tensor(tn(LLM_TENSOR_NEXTN_HIDDEN_PROJ, "weight", il), { n_embd, n_embd }, mtp_flags);
        layer.nextn.hc_head_norm = create_tensor(tn(LLM_TENSOR_NEXTN_HC_HEAD_NORM, "weight", il), { n_hc_embd }, mtp_flags);
        layer.nextn.hc_head_down = create_tensor(tn(LLM_TENSOR_NEXTN_HC_HEAD_DOWN, "weight", il), { n_hc_embd, n_hc_rank }, mtp_flags);
        layer.nextn.hc_head_up = create_tensor(tn(LLM_TENSOR_NEXTN_HC_HEAD_UP, "weight", il), { n_hc_rank, n_hc_embd }, mtp_flags);
    }
}

std::unique_ptr<llm_graph_context> llama_model_qwen4exp::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_qwen4exp::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_build_delta_net_base(params), model(model) {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        build_mtp();
        return;
    }

    const int64_t n_hc = hparams.qwen4_hc_count;

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    ggml_tensor * inpL = build_inp_embd(model.tok_embd);
    inpL = ggml_reshape_3d(ctx0, inpL, n_embd, 1, n_tokens);
    inpL = ggml_repeat_4d(ctx0, inpL, n_embd, n_hc, n_tokens, 1);
    inpL = ggml_reshape_2d(ctx0, inpL, n_embd * n_hc, n_tokens);
    cb(inpL, "hc_init", -1);

    auto * inp = build_inp_mem_qwen4();

    uint32_t qsa_ratio = 0;
    for (int il = 0; il < n_layer; ++il) {
        if (!hparams.is_recr(il)) {
            qsa_ratio = hparams.qwen4_compress_ratios[il];
            break;
        }
    }
    GGML_ASSERT(qsa_ratio > 0);

    llm_graph_input_qwen4_qsa * qsa = build_qsa_input(inp, qsa_ratio);

    ggml_tensor * inp_pos = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();
    const bool crop_last_layer = inp_out_ids && (!cparams.embeddings_nextn || cparams.embeddings_nextn_masked);

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];
        res->t_layer_inp[il] = inpL;

        if (il == hparams.qwen4_ple_layer) {
            inpL = ggml_add(ctx0, inpL, build_ple(inp->get_ple(), inp->get_ple_ids(), inpL, il));
            cb(inpL, "ple_out", il);
        }

        auto [cur, residual, injection] = build_hc_mix(
                inpL,
                layer.hc_attn_norm,
                layer.hc_attn_down,
                layer.hc_attn_up,
                layer.hc_attn_inject,
                il);
        cb(cur, "hc_attn_mix", il);

        if (hparams.is_recr(il)) {
            cur = build_layer_attn_linear(inp->get_gdn(), cur, il);
        } else {
            cur = build_layer_attn(inp->get_attn(), qsa, cur, inp_pos, sections, il);
        }

        if (il == n_layer - 1 && crop_last_layer) {
            cur = ggml_get_rows(ctx0, cur, inp_out_ids);
            residual = ggml_get_rows(ctx0, residual, inp_out_ids);
            injection = ggml_get_rows(ctx0, injection, inp_out_ids);
        }

        inpL = build_hc_combine(cur, residual, injection, il);
        cb(inpL, "hc_attn_out", il);

        std::tie(cur, residual, injection) = build_hc_mix(
                inpL,
                layer.hc_ffn_norm,
                layer.hc_ffn_down,
                layer.hc_ffn_up,
                layer.hc_ffn_inject,
                il);
        cb(cur, "hc_ffn_mix", il);

        cur = build_layer_ffn(cur, il);
        inpL = build_hc_combine(cur, residual, injection, il);

        const int64_t n_layer_tokens = inpL->ne[1];
        inpL = ggml_reshape_2d(ctx0, inpL, n_embd, n_hc * n_layer_tokens);
        inpL = build_cvec(inpL, il);
        inpL = ggml_reshape_2d(ctx0, inpL, n_embd * n_hc, n_layer_tokens);
        cb(inpL, "l_out", il);
    }

    if (cparams.embeddings_nextn) {
        cb(inpL, "h_nextn", -1);
        res->t_h_nextn = inpL;
    }

    ggml_tensor * cur = build_hc_head(inpL, model.hc_head_norm, model.hc_head_down, model.hc_head_up, -1);
    cb(cur, "hc_head", -1);

    if (!crop_last_layer && inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cb(cur, "result_embd", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur, model.output_s);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

ggml_tensor * llama_model_qwen4exp::graph::build_hc_norm(
        ggml_tensor * input,
        ggml_tensor * weight,
        int           il) {
    const int64_t n_hc = hparams.qwen4_hc_count;
    const int64_t n_tokens_cur = input->ne[1];

    ggml_tensor * cur = ggml_reshape_3d(ctx0, input, n_embd, n_hc, n_tokens_cur);
    cur = ggml_rms_norm(ctx0, cur, hparams.f_norm_rms_eps);

    weight = ggml_reshape_2d(ctx0, weight, n_embd, n_hc);
    cur = ggml_mul(ctx0, cur, weight);
    cur = ggml_reshape_2d(ctx0, cur, n_embd * n_hc, n_tokens_cur);
    cb(cur, "hc_norm", il);
    return cur;
}

std::tuple<ggml_tensor *, ggml_tensor *, ggml_tensor *> llama_model_qwen4exp::graph::build_hc_mix(
        ggml_tensor * input,
        ggml_tensor * norm,
        ggml_tensor * down,
        ggml_tensor * up,
        ggml_tensor * inject,
        int           il) {
    const int64_t n_hc = hparams.qwen4_hc_count;
    const int64_t n_tokens_cur = input->ne[1];

    ggml_tensor * normalized = build_hc_norm(input, norm, il);
    ggml_tensor * mix = build_lora_mm(down, normalized);
    mix = ggml_scale(ctx0, mix, 1.0f / n_hc);
    mix = ggml_silu(ctx0, mix);
    mix = build_lora_mm(up, mix);
    mix = ggml_sigmoid(ctx0, mix);
    cb(mix, "hc_gate", il);
    mix = ggml_mul(ctx0, mix, normalized);
    mix = ggml_reshape_3d(ctx0, mix, n_embd, n_hc, n_tokens_cur);
    mix = qwen4_hc_mean(ctx0, mix);

    ggml_tensor * injection = build_lora_mm(inject, normalized);
    injection = ggml_scale(ctx0, injection, 1.0f / n_hc);
    injection = ggml_sigmoid(ctx0, injection);
    injection = ggml_scale(ctx0, injection, 2.0f);

    return { mix, input, injection };
}

ggml_tensor * llama_model_qwen4exp::graph::build_hc_combine(
        ggml_tensor * output,
        ggml_tensor * residual,
        ggml_tensor * injection,
        int           il) {
    const int64_t n_hc = hparams.qwen4_hc_count;
    const int64_t n_tokens_cur = output->ne[1];

    output = ggml_reshape_3d(ctx0, output, n_embd, 1, n_tokens_cur);
    output = ggml_repeat_4d(ctx0, output, n_embd, n_hc, n_tokens_cur, 1);
    injection = ggml_reshape_3d(ctx0, injection, 1, n_hc, n_tokens_cur);
    output = ggml_mul(ctx0, output, injection);
    output = ggml_reshape_2d(ctx0, output, n_embd * n_hc, n_tokens_cur);
    output = ggml_add(ctx0, residual, output);
    cb(output, "hc_combine", il);
    return output;
}

ggml_tensor * llama_model_qwen4exp::graph::build_hc_head(
        ggml_tensor * input,
        ggml_tensor * norm,
        ggml_tensor * down,
        ggml_tensor * up,
        int           il) {
    const int64_t n_hc = hparams.qwen4_hc_count;
    const int64_t n_tokens_cur = input->ne[1];

    ggml_tensor * normalized = build_hc_norm(input, norm, il);
    ggml_tensor * mix = build_lora_mm(down, normalized);
    mix = ggml_scale(ctx0, mix, 1.0f / n_hc);
    mix = ggml_silu(ctx0, mix);
    mix = build_lora_mm(up, mix);
    mix = ggml_sigmoid(ctx0, mix);
    mix = ggml_mul(ctx0, mix, normalized);
    mix = ggml_reshape_3d(ctx0, mix, n_embd, n_hc, n_tokens_cur);
    return qwen4_hc_mean(ctx0, mix);
}

llm_graph_input_qwen4_qsa * llama_model_qwen4exp::graph::build_qsa_input(
        llm_graph_input_mem_qwen4 * inp,
        uint32_t                    ratio) {
    const int64_t n_kv = inp->get_attn()->get_kq_mask()->ne[0];
    if (n_kv <= (int64_t) hparams.indexer_top_k + ratio - 1) {
        return nullptr;
    }

    const uint32_t block_topk = hparams.indexer_top_k / ratio;
    const int64_t n_blocks = n_kv / ratio;
    auto qsa_input = std::make_unique<llm_graph_input_qwen4_qsa>(
            static_cast<const llama_memory_qwen4_context *>(mctx),
            inp->get_attn()->get_kq_mask(), ratio, block_topk);
    qsa_input->block_cells = ggml_new_tensor_3d(ctx0, GGML_TYPE_I32, ratio, n_blocks, n_tokens);
    qsa_input->block_pos = ggml_new_tensor_3d(ctx0, GGML_TYPE_I32, n_blocks, hparams.n_pos_per_embd(), n_tokens);
    qsa_input->block_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_blocks, n_tokens);
    qsa_input->selected = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_kv, n_tokens);
    ggml_set_input(qsa_input->block_cells);
    ggml_set_input(qsa_input->block_pos);
    ggml_set_input(qsa_input->block_mask);
    ggml_set_input(qsa_input->selected);
    ggml_set_name(qsa_input->block_cells, "qsa_block_cells");
    ggml_set_name(qsa_input->block_pos, "qsa_block_pos");
    ggml_set_name(qsa_input->block_mask, "qsa_block_mask");
    ggml_set_name(qsa_input->selected, "qsa_selected_base");
    return (llm_graph_input_qwen4_qsa *) res->add_input(std::move(qsa_input));
}

std::pair<ggml_tensor *, ggml_tensor *> llama_model_qwen4exp::graph::build_qkvz(
        ggml_tensor * input,
        int           il) {
    const int64_t n_seqs = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    ggml_tensor * qkv = build_lora_mm(model.layers[il].wqkv, input, model.layers[il].wqkv_s);
    qkv = ggml_reshape_3d(ctx0, qkv, qkv->ne[0], n_seq_tokens, n_seqs);
    cb(qkv, "linear_attn_qkv", il);

    ggml_tensor * gate = build_lora_mm(model.layers[il].wqkv_gate, input, model.layers[il].wqkv_gate_s);
    cb(gate, "linear_attn_gate", il);
    return { qkv, gate };
}

ggml_tensor * llama_model_qwen4exp::graph::build_norm_gated(
        ggml_tensor * input,
        ggml_tensor * weights,
        ggml_tensor * gate,
        int           il) {
    ggml_tensor * normalized = build_norm(input, weights, nullptr, LLM_NORM_RMS, il);
    return ggml_mul(ctx0, normalized, ggml_sigmoid(ctx0, gate));
}

ggml_tensor * llama_model_qwen4exp::graph::build_layer_attn(
        llm_graph_input_attn_kv_msa * inp,
        llm_graph_input_qwen4_qsa *   qsa,
        ggml_tensor *                 cur,
        ggml_tensor *                 inp_pos,
        int *                         sections,
        int                           il) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    ggml_tensor * index_k = build_lora_mm(model.layers[il].index_k_proj, cur);
    index_k = ggml_reshape_3d(ctx0, index_k, hparams.indexer_head_size, 1, n_tokens);
    cb(index_k, "index_Kraw", il);
    ggml_build_forward_expand(gf, inp->mctx_msa->get_idx()->cpy_k(ctx0, index_k, inp->get_k_idxs_idx(), il));

    ggml_tensor * q_full = build_lora_mm(model.layers[il].wq, cur, model.layers[il].wq_s);
    ggml_tensor * q = ggml_view_3d(ctx0, q_full, n_embd_head, n_head, n_tokens,
            ggml_element_size(q_full) * n_embd_head * 2,
            ggml_element_size(q_full) * n_embd_head * 2 * n_head,
            0);
    q = build_norm(q, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);

    ggml_tensor * gate = ggml_view_3d(ctx0, q_full, n_embd_head, n_head, n_tokens,
            ggml_element_size(q_full) * n_embd_head * 2,
            ggml_element_size(q_full) * n_embd_head * 2 * n_head,
            ggml_element_size(q_full) * n_embd_head);
    gate = ggml_cont_2d(ctx0, gate, n_embd_head * n_head, n_tokens);

    ggml_tensor * k = build_lora_mm(model.layers[il].wk, cur, model.layers[il].wk_s);
    k = ggml_reshape_3d(ctx0, k, n_embd_head, n_head_kv, n_tokens);
    k = build_norm(k, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);

    ggml_tensor * v = build_lora_mm(model.layers[il].wv, cur, model.layers[il].wv_s);
    v = ggml_reshape_3d(ctx0, v, n_embd_head, n_head_kv, n_tokens);

    if (std::accumulate(sections, sections + 4, 0) <= n_embd_head) {
        q = ggml_rope_multi(ctx0, q, inp_pos, nullptr, n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
        k = ggml_rope_multi(ctx0, k, inp_pos, nullptr, n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
    } else {
        ggml_tensor * text_pos = ggml_view_1d(ctx0, inp_pos, n_tokens, 0);
        q = ggml_rope_ext(ctx0, q, text_pos, nullptr, n_rot, LLAMA_ROPE_TYPE_NORM, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
        k = ggml_rope_ext(ctx0, k, text_pos, nullptr, n_rot, LLAMA_ROPE_TYPE_NORM, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
    }
    cb(q, "Qcur", il);
    cb(k, "Kcur", il);
    cb(v, "Vcur", il);

    inp->self_kq_mask_cnv = qsa ? build_qsa_mask(inp, qsa, cur, inp_pos, sections, il) : inp->self_kq_mask;

    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;
    cur = build_attn(inp, nullptr, nullptr, nullptr, q, k, v, nullptr, nullptr, nullptr, kq_scale, il);
    cur = ggml_mul(ctx0, cur, ggml_sigmoid(ctx0, gate));
    cur = build_lora_mm(model.layers[il].wo, cur, model.layers[il].wo_s);
    cb(cur, "attn_out", il);
    return cur;
}

ggml_tensor * llama_model_qwen4exp::graph::build_qsa_mask(
        llm_graph_input_attn_kv_msa * inp,
        llm_graph_input_qwen4_qsa *   qsa,
        ggml_tensor *                 cur,
        ggml_tensor *                 inp_pos,
        int *                         sections,
        int                           il) {
    const int64_t index_dim = hparams.indexer_head_size;
    const int64_t index_heads = hparams.indexer_n_head;
    const int64_t ratio = hparams.qwen4_compress_ratios[il];
    const int64_t block_topk = hparams.indexer_top_k / ratio;
    const int64_t n_blocks = qsa->block_cells->ne[1];
    const int64_t n_kv = qsa->selected->ne[0];
    const int64_t n_pos = hparams.n_pos_per_embd();

    ggml_tensor * index_q = build_lora_mm(model.layers[il].index_q_proj, cur);
    index_q = ggml_reshape_3d(ctx0, index_q, index_dim, index_heads, n_tokens);
    index_q = build_norm(index_q, model.layers[il].index_q_norm, nullptr, LLM_NORM_RMS, il);

    if (std::accumulate(sections, sections + 4, 0) <= index_dim) {
        index_q = ggml_rope_multi(ctx0, index_q, inp_pos, nullptr, n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
    } else {
        index_q = ggml_rope_ext(ctx0, index_q, ggml_view_1d(ctx0, inp_pos, n_tokens, 0), nullptr, n_rot, LLAMA_ROPE_TYPE_NORM, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
    }
    cb(index_q, "index_Q", il);

    const auto * mctx_idx = inp->mctx_msa->get_idx();
    ggml_tensor * index_k_cache = mctx_idx->get_k(ctx0, il);
    const int64_t n_stream = index_k_cache->ne[3];
    GGML_ASSERT(n_tokens % n_stream == 0);
    const int64_t n_tps = n_tokens / n_stream;
    const int64_t selected_per_query = ratio * block_topk;
    const ggml_type activation_type = mctx_idx->type_k();
    ggml_tensor * index_k_norm = model.layers[il].index_k_norm;
    std::vector<ggml_tensor *> selected_streams;
    selected_streams.reserve(n_stream);

    for (int64_t is = 0; is < n_stream; ++is) {
        ggml_tensor * cache = ggml_view_2d(
                ctx0, index_k_cache, index_dim, n_kv,
                index_k_cache->nb[2], is * index_k_cache->nb[3]);
        ggml_tensor * block_cells = ggml_view_3d(
                ctx0, qsa->block_cells, ratio, n_blocks, n_tps,
                qsa->block_cells->nb[1], qsa->block_cells->nb[2], is * n_tps * qsa->block_cells->nb[2]);
        ggml_tensor * block_keys = ggml_get_rows(
                ctx0, cache, ggml_reshape_1d(ctx0, block_cells, ratio * n_blocks * n_tps));
        block_keys = ggml_reshape_4d(ctx0, block_keys, index_dim, ratio, n_blocks, n_tps);
        block_keys = ggml_cont(ctx0, ggml_transpose(ctx0, block_keys));
        block_keys = ggml_mean(ctx0, block_keys);
        block_keys = ggml_cont(ctx0, ggml_transpose(ctx0, block_keys));
        block_keys = ggml_reshape_3d(ctx0, block_keys, index_dim, 1, n_blocks * n_tps);

        if (block_keys->type != activation_type) {
            block_keys = ggml_cast(ctx0, block_keys, activation_type);
            block_keys = ggml_cast(ctx0, block_keys, GGML_TYPE_F32);
        }
        block_keys = build_norm(block_keys, index_k_norm, nullptr, LLM_NORM_RMS, il);

        ggml_tensor * block_pos = ggml_view_3d(
                ctx0, qsa->block_pos, n_blocks, n_pos, n_tps,
                qsa->block_pos->nb[1], qsa->block_pos->nb[2], is * n_tps * qsa->block_pos->nb[2]);
        if (std::accumulate(sections, sections + 4, 0) <= index_dim) {
            block_keys = ggml_rope_multi(ctx0, block_keys, ggml_reshape_1d(ctx0, block_pos, n_blocks * n_pos * n_tps), nullptr, n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
        } else {
            block_pos = ggml_view_2d(ctx0, block_pos, n_blocks, n_tps, block_pos->nb[2], 0);
            block_keys = ggml_rope_ext(ctx0, block_keys, ggml_reshape_1d(ctx0, ggml_cont(ctx0, block_pos), n_blocks * n_tps), nullptr, n_rot, LLAMA_ROPE_TYPE_NORM, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
        }
        cb(block_keys, "index_Kblock", il);

        block_keys = ggml_reshape_4d(ctx0, block_keys, index_dim, n_blocks, 1, n_tps);
        ggml_tensor * query = ggml_view_3d(
                ctx0, index_q, index_dim, index_heads, n_tps,
                index_q->nb[1], index_q->nb[2], is * n_tps * index_q->nb[2]);
        query = ggml_reshape_4d(ctx0, query, index_dim, index_heads, 1, n_tps);
        ggml_tensor * scores = ggml_mul_mat(ctx0, block_keys, query);
        ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
        scores = ggml_relu(ctx0, scores);
        scores = ggml_sum_rows(ctx0, ggml_cont(ctx0, ggml_permute(ctx0, scores, 1, 0, 2, 3)));
        scores = ggml_scale(ctx0, ggml_reshape_2d(ctx0, scores, n_blocks, n_tps), 1.0f / sqrtf(float(index_dim)));
        ggml_tensor * block_mask = ggml_view_2d(
                ctx0, qsa->block_mask, n_blocks, n_tps,
                qsa->block_mask->nb[1], is * n_tps * qsa->block_mask->nb[1]);
        scores = ggml_add(ctx0, scores, block_mask);
        cb(scores, "index_scores", il);

        ggml_tensor * top_blocks = ggml_argsort_top_k(ctx0, scores, block_topk);
        ggml_tensor * top_cells = ggml_get_rows(ctx0, block_cells, top_blocks);
        top_cells = ggml_reshape_2d(ctx0, top_cells, selected_per_query, n_tps);
        cb(top_cells, "index_selected_cells", il);

        ggml_tensor * base_selected = ggml_view_2d(
                ctx0, qsa->selected, n_kv, n_tps,
                qsa->selected->nb[1], is * n_tps * qsa->selected->nb[1]);
        base_selected = ggml_reshape_3d(ctx0, base_selected, 1, n_kv, n_tps);
        ggml_tensor * selected_top = ggml_fill(ctx0, base_selected, 0.0f);
        ggml_tensor * ones = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, selected_per_query, n_tps);
        ones = ggml_fill(ctx0, ones, 1.0f);
        selected_top = ggml_set_rows(ctx0, selected_top, ones, top_cells);
        ggml_tensor * selected_stream = ggml_clamp(ctx0, ggml_add(ctx0, base_selected, selected_top), 0.0f, 1.0f);
        selected_streams.push_back(ggml_reshape_2d(ctx0, selected_stream, n_kv, n_tps));
    }

    ggml_tensor * selected = selected_streams[0];
    for (int64_t is = 1; is < n_stream; ++is) {
        selected = ggml_concat(ctx0, selected, selected_streams[is], 1);
    }
    selected = ggml_scale_bias(ctx0, selected, 1e30f, -1e30f);

    ggml_tensor * base_mask = ggml_reshape_2d(ctx0, qsa->kq_mask, n_kv, n_tokens);
    if (base_mask->type != GGML_TYPE_F32) {
        base_mask = ggml_cast(ctx0, base_mask, GGML_TYPE_F32);
    }
    ggml_tensor * mask = ggml_add(ctx0, base_mask, selected);
    if (cparams.flash_attn) {
        mask = ggml_cast(ctx0, mask, GGML_TYPE_F16);
    }
    cb(mask, "qsa_mask", il);
    return mask;
}

ggml_tensor * llama_model_qwen4exp::graph::build_layer_attn_linear(
        llm_graph_input_rs * inp,
        ggml_tensor *        cur,
        int                  il) {
    const auto * mctx_cur = inp->mctx;
    const int64_t d_inner = hparams.ssm_d_inner;
    const int64_t n_seqs = ubatch.n_seqs;
    const int64_t head_k_dim = hparams.ssm_d_state;
    const int64_t n_k_heads = hparams.ssm_n_group;
    const int64_t n_v_heads = hparams.ssm_dt_rank;
    const int64_t head_v_dim = d_inner / n_v_heads;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);

    auto [qkv, z] = build_qkvz(cur, il);

    ggml_tensor * beta = build_lora_mm(model.layers[il].ssm_beta, cur, model.layers[il].ssm_beta_s);
    beta = ggml_reshape_4d(ctx0, ggml_sigmoid(ctx0, beta), 1, n_v_heads, n_seq_tokens, n_seqs);
    cb(beta, "linear_attn_beta", il);

    ggml_tensor * alpha = build_lora_mm(model.layers[il].ssm_alpha, cur, model.layers[il].ssm_alpha_s);
    alpha = ggml_reshape_3d(ctx0, alpha, n_v_heads, n_seq_tokens, n_seqs);
    alpha = ggml_softplus(ctx0, ggml_add(ctx0, alpha, model.layers[il].ssm_dt));
    ggml_tensor * decay = ggml_mul(ctx0, alpha, model.layers[il].ssm_a);
    decay = ggml_reshape_4d(ctx0, decay, 1, n_v_heads, n_seq_tokens, n_seqs);
    cb(decay, "linear_attn_decay", il);

    ggml_tensor * conv_states = mctx_cur->get_r_l(il);
    ggml_tensor * ssm_states = mctx_cur->get_s_l(il);
    ggml_tensor * conv_kernel = model.layers[il].ssm_conv1d;
    const int64_t conv_kernel_size = conv_kernel->ne[0];
    const int64_t conv_channels = d_inner + 2 * n_k_heads * head_k_dim;

    ggml_tensor * conv_input = build_conv_state(inp, conv_states, qkv, conv_kernel_size, conv_channels, il);
    ggml_tensor * state = build_rs(inp, ssm_states, hparams.n_embd_s(), n_seqs);
    state = ggml_reshape_4d(ctx0, state, head_v_dim, head_v_dim, n_v_heads, n_seqs);

    ggml_tensor * qkv_conv = ggml_silu(ctx0, ggml_ssm_conv(ctx0, conv_input, conv_kernel));
    cb(qkv_conv, "linear_attn_conv", il);
    const int64_t qkv_dim = 2 * head_k_dim * n_k_heads + d_inner;
    const int64_t nb1_qkv = ggml_row_size(qkv_conv->type, qkv_dim);

    ggml_tensor * q = ggml_view_4d(ctx0, qkv_conv, head_k_dim, n_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(qkv_conv->type, head_k_dim), nb1_qkv, nb1_qkv * n_seq_tokens, 0);
    ggml_tensor * k = ggml_view_4d(ctx0, qkv_conv, head_k_dim, n_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(qkv_conv->type, head_k_dim), nb1_qkv, nb1_qkv * n_seq_tokens,
            head_k_dim * n_k_heads * ggml_element_size(qkv_conv));
    ggml_tensor * v = ggml_view_4d(ctx0, qkv_conv, head_v_dim, n_v_heads, n_seq_tokens, n_seqs,
            ggml_row_size(qkv_conv->type, head_v_dim), nb1_qkv, nb1_qkv * n_seq_tokens,
            ggml_row_size(qkv_conv->type, 2 * head_k_dim * n_k_heads));

    q = qwen4_l2_norm(ctx0, q);
    k = qwen4_l2_norm(ctx0, k);
    cb(q, "linear_attn_q", il);
    cb(k, "linear_attn_k", il);
    cb(v, "linear_attn_v", il);

    if (n_k_heads != n_v_heads && (!cparams.fused_gdn_ar || !cparams.fused_gdn_ch)) {
        GGML_ASSERT(n_v_heads % n_k_heads == 0);
        q = ggml_repeat_4d(ctx0, q, head_k_dim, n_v_heads, n_seq_tokens, n_seqs);
        k = ggml_repeat_4d(ctx0, k, head_k_dim, n_v_heads, n_seq_tokens, n_seqs);
    }

    ggml_tensor * output = build_recurrent_attn(inp, ssm_states, q, k, v, decay, beta, state, il);
    cb(output, "linear_attn_gdn", il);
    z = ggml_reshape_4d(ctx0, z, head_v_dim, n_v_heads, n_seq_tokens, n_seqs);
    output = build_norm_gated(output, model.layers[il].ssm_norm, z, il);
    cb(output, "linear_attn_norm", il);
    output = ggml_reshape_3d(ctx0, output, d_inner, n_seq_tokens, n_seqs);

    cur = build_lora_mm(model.layers[il].ssm_out, output, model.layers[il].ssm_out_s);
    cur = ggml_reshape_2d(ctx0, cur, n_embd, n_seq_tokens * n_seqs);
    cb(cur, "linear_attn_out", il);
    return cur;
}

ggml_tensor * llama_model_qwen4exp::graph::build_layer_ffn(ggml_tensor * cur, int il) {
    const auto & layer = model.layers[il];
    ggml_tensor * moe = build_moe_ffn(
            cur,
            layer.ffn_gate_inp,
            layer.ffn_up_exps,
            layer.ffn_gate_exps,
            layer.ffn_down_exps,
            nullptr,
            n_expert,
            n_expert_used,
            LLM_FFN_SILU,
            true,
            hparams.expert_weights_scale,
            LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
            il,
            nullptr,
            layer.ffn_gate_up_exps,
            layer.ffn_up_exps_s,
            layer.ffn_gate_exps_s,
            layer.ffn_down_exps_s);
    cb(moe, "ffn_moe_out", il);

    ggml_tensor * shared = build_ffn(
            cur,
            layer.ffn_up_shexp, nullptr, layer.ffn_up_shexp_s,
            layer.ffn_gate_shexp, nullptr, layer.ffn_gate_shexp_s,
            layer.ffn_down_shexp, nullptr, layer.ffn_down_shexp_s,
            nullptr,
            LLM_FFN_SILU,
            LLM_FFN_PAR,
            il);
    ggml_tensor * shared_gate = ggml_sigmoid(ctx0, build_lora_mm(layer.ffn_gate_inp_shexp, cur));
    shared = ggml_mul(ctx0, shared, shared_gate);
    cur = ggml_add(ctx0, moe, shared);
    cb(cur, "ffn_out", il);
    return cur;
}

ggml_tensor * llama_model_qwen4exp::graph::build_ple(
        llm_graph_input_rs * inp,
        ggml_tensor *        ple_ids,
        ggml_tensor *        input,
        int                  il) {
    const auto & layer = model.layers[il];
    const int64_t n_hc = hparams.qwen4_hc_count;
    const int64_t n_ple_heads = (hparams.qwen4_ple_ngram - 1) * hparams.qwen4_ple_heads;
    const int64_t n_ple_embd = hparams.n_embd_per_layer * n_ple_heads;

    ggml_tensor * embedding = ggml_get_rows(ctx0, model.per_layer_tok_embd, ggml_reshape_1d(ctx0, ple_ids, n_ple_heads * n_tokens));
    embedding = ggml_reshape_2d(ctx0, embedding, n_ple_embd, n_tokens);
    cb(embedding, "ple_embedding", il);

    ggml_tensor * key = build_lora_mm(layer.ple_key, embedding);
    key = build_hc_norm(key, layer.ple_norm_key, il);
    key = ggml_reshape_3d(ctx0, key, n_embd, n_hc, n_tokens);
    cb(key, "ple_key", il);

    ggml_tensor * query = build_hc_norm(input, layer.ple_norm_query, il);
    query = ggml_reshape_3d(ctx0, query, n_embd, n_hc, n_tokens);
    cb(query, "ple_query", il);

    ggml_tensor * gate = ggml_sum_rows(ctx0, ggml_mul(ctx0, key, query));
    gate = ggml_scale(ctx0, gate, 1.0f / sqrtf(float(n_embd)));
    gate = ggml_mul(ctx0,
            ggml_sgn(ctx0, gate),
            ggml_sqrt(ctx0, ggml_clamp(ctx0, ggml_abs(ctx0, gate), 1e-6f, std::numeric_limits<float>::max())));
    gate = ggml_sigmoid(ctx0, gate);
    cb(gate, "ple_gate", il);

    ggml_tensor * value = build_lora_mm(layer.ple_value, embedding);
    value = ggml_reshape_3d(ctx0, value, n_embd, 1, n_tokens);
    value = ggml_repeat_4d(ctx0, value, n_embd, n_hc, n_tokens, 1);
    ggml_tensor * gated = ggml_mul(ctx0, value, gate);
    gated = ggml_reshape_2d(ctx0, gated, n_embd * n_hc, n_tokens);
    cb(gated, "ple_gated", il);

    ggml_tensor * conv_input = build_hc_norm(gated, layer.ple_norm_conv, il);
    ggml_tensor * conv = build_ple_conv(inp, conv_input, layer.ple_conv1d, il);
    conv = ggml_silu(ctx0, conv);
    cb(conv, "ple_conv", il);

    return ggml_add(ctx0, gated, conv);
}

ggml_tensor * llama_model_qwen4exp::graph::build_ple_conv(
        llm_graph_input_rs * inp,
        ggml_tensor *        input,
        ggml_tensor *        weight,
        int                  il) {
    const auto * mctx_cur = inp->mctx;
    const int64_t n_hc_embd = n_embd * hparams.qwen4_hc_count;
    const int64_t history_len = (hparams.qwen4_ple_conv - 1) * hparams.qwen4_ple_ngram;
    const int64_t state_size = n_hc_embd * history_len;
    const int64_t n_seqs = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(n_tokens == n_seq_tokens * n_seqs);

    ggml_tensor * states_all = mctx_cur->get_r_l(il);
    ggml_tensor * states = build_rs(inp, states_all, state_size, n_seqs);
    states = ggml_reshape_3d(ctx0, states, history_len, n_hc_embd, n_seqs);

    input = ggml_reshape_3d(ctx0, input, n_hc_embd, n_seq_tokens, n_seqs);
    input = ggml_cont(ctx0, ggml_transpose(ctx0, input));
    ggml_tensor * conv_input = ggml_concat(ctx0, states, input, 0);
    cb(conv_input, "ple_conv_input", il);

    const int64_t width = conv_input->ne[0];
    const int64_t kv_head = mctx_cur->get_head();
    const int64_t mem_size = mctx_cur->get_size();
    const int64_t snapshots = cparams.n_rs_seq == 0 ? 1 : cparams.n_rs_seq + 1;

    for (int64_t t = 1; t <= snapshots; ++t) {
        const int64_t source = std::max<int64_t>(0, width - history_len - snapshots + t);
        const int64_t slot = snapshots - t;
        ggml_tensor * state_last = ggml_view_3d(
                ctx0, conv_input, history_len, n_hc_embd, n_seqs,
                conv_input->nb[1], conv_input->nb[2], ggml_row_size(conv_input->type, source));
        ggml_tensor * state_update = ggml_view_2d(
                ctx0, states_all, state_size, n_seqs, states_all->nb[1],
                (slot * mem_size + kv_head) * states_all->nb[1]);
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, state_last, state_update));
    }

    std::vector<ggml_tensor *> outputs;
    outputs.reserve(n_seqs);
    weight = ggml_reshape_3d(ctx0, weight, hparams.qwen4_ple_conv, 1, n_hc_embd);
    for (int64_t is = 0; is < n_seqs; ++is) {
        ggml_tensor * seq_input = ggml_view_2d(
                ctx0, conv_input, width, n_hc_embd, conv_input->nb[1], is * conv_input->nb[2]);
        ggml_tensor * output = ggml_conv_1d_dw(
                ctx0, weight, seq_input, 1, 0, hparams.qwen4_ple_ngram);
        output = ggml_cont(ctx0, ggml_transpose(ctx0, output));
        outputs.push_back(output);
    }

    ggml_tensor * result = outputs[0];
    for (int64_t is = 1; is < n_seqs; ++is) {
        result = ggml_concat(ctx0, result, outputs[is], 1);
    }
    return ggml_reshape_2d(ctx0, result, n_hc_embd, n_tokens);
}

void llama_model_qwen4exp::graph::build_mtp() {
    GGML_ASSERT(hparams.n_layer_nextn == 1 && "Qwen4 MTP requires one NextN layer");
    GGML_ASSERT(cparams.nextn_layer_offset >= 0 &&
            cparams.nextn_layer_offset < (int) hparams.n_layer_nextn &&
            "nextn_layer_offset out of range");
    GGML_ASSERT(ubatch.token && "Qwen4 MTP requires token input");

    const int64_t n_hc = hparams.qwen4_hc_count;
    const int64_t n_hc_embd = n_hc * n_embd;
    GGML_ASSERT(hparams.n_embd_nextn() == (uint32_t) n_hc_embd && "Qwen4 MTP hidden width mismatch");

    const int il = hparams.n_layer() + cparams.nextn_layer_offset;
    const auto & layer = model.layers[il];
    GGML_ASSERT(layer.nextn.enorm && "Qwen4 MTP is missing nextn.enorm");
    GGML_ASSERT(layer.nextn.hnorm && "Qwen4 MTP is missing nextn.hnorm");
    GGML_ASSERT(layer.nextn.embed_proj && "Qwen4 MTP is missing nextn.embed_proj");
    GGML_ASSERT(layer.nextn.hidden_proj && "Qwen4 MTP is missing nextn.hidden_proj");
    GGML_ASSERT(layer.nextn.hc_head_norm && "Qwen4 MTP is missing nextn.hc_head_norm");
    GGML_ASSERT(layer.nextn.hc_head_down && "Qwen4 MTP is missing nextn.hc_head_down");
    GGML_ASSERT(layer.nextn.hc_head_up && "Qwen4 MTP is missing nextn.hc_head_up");

    auto inp = std::make_unique<llm_graph_input_embd_h>(n_hc_embd);
    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);
    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_hc_embd, n_tokens);
    ggml_set_input(inp->embd);
    inp->h = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_hc_embd, n_tokens);
    ggml_set_input(inp->h);
    ggml_set_name(inp->h, "mtp_h_input");

    ggml_tensor * tok_embd = ggml_get_rows(ctx0, model.tok_embd, inp->tokens);
    cb(tok_embd, "mtp_tok_embd", il);

    ggml_tensor * hidden = build_norm(inp->h, layer.nextn.hnorm, nullptr, LLM_NORM_RMS, il);
    hidden = ggml_reshape_3d(ctx0, hidden, n_embd, n_hc, n_tokens);
    hidden = build_lora_mm(layer.nextn.hidden_proj, hidden);
    cb(hidden, "mtp_hidden_proj", il);

    ggml_tensor * embedded = build_norm(tok_embd, layer.nextn.enorm, nullptr, LLM_NORM_RMS, il);
    embedded = build_lora_mm(layer.nextn.embed_proj, embedded);
    embedded = ggml_reshape_3d(ctx0, embedded, n_embd, 1, n_tokens);
    embedded = ggml_repeat_4d(ctx0, embedded, n_embd, n_hc, n_tokens, 1);
    cb(embedded, "mtp_embed_proj", il);

    ggml_tensor * inpL = ggml_add(ctx0, hidden, embedded);
    inpL = ggml_reshape_2d(ctx0, inpL, n_hc_embd, n_tokens);
    cb(inpL, "mtp_input_fused", il);

    res->add_input(std::move(inp));

    auto * inp_mem = build_inp_mem_qwen4(false);
    llm_graph_input_qwen4_qsa * qsa = build_qsa_input(inp_mem, hparams.qwen4_compress_ratios[il]);

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);
    ggml_tensor * inp_pos = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();
    const bool crop_layer = inp_out_ids && (!cparams.embeddings_nextn || cparams.embeddings_nextn_masked);

    auto [cur, residual, injection] = build_hc_mix(
            inpL,
            layer.hc_attn_norm,
            layer.hc_attn_down,
            layer.hc_attn_up,
            layer.hc_attn_inject,
            il);
    cb(cur, "mtp_hc_attn_mix", il);

    cur = build_layer_attn(inp_mem->get_attn(), qsa, cur, inp_pos, sections, il);
    if (crop_layer) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
        residual = ggml_get_rows(ctx0, residual, inp_out_ids);
        injection = ggml_get_rows(ctx0, injection, inp_out_ids);
    }
    inpL = build_hc_combine(cur, residual, injection, il);
    cb(inpL, "mtp_hc_attn_out", il);

    std::tie(cur, residual, injection) = build_hc_mix(
            inpL,
            layer.hc_ffn_norm,
            layer.hc_ffn_down,
            layer.hc_ffn_up,
            layer.hc_ffn_inject,
            il);
    cb(cur, "mtp_hc_ffn_mix", il);

    cur = build_layer_ffn(cur, il);
    inpL = build_hc_combine(cur, residual, injection, il);
    cb(inpL, "mtp_l_out", il);

    if (cparams.embeddings_nextn) {
        cb(inpL, "h_nextn", -1);
        res->t_h_nextn = inpL;
    }

    if (!crop_layer && inp_out_ids) {
        inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
    }

    cur = build_hc_head(
            inpL,
            layer.nextn.hc_head_norm,
            layer.nextn.hc_head_down,
            layer.nextn.hc_head_up,
            il);
    cb(cur, "mtp_hc_head", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur, model.output_s);
    cb(cur, "result_output", -1);
    res->t_logits = cur;
    ggml_build_forward_expand(gf, cur);
}
