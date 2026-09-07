#include "ggml-metal-fusion.h"

#include "ggml-backend-impl.h"
#include "ggml-metal-device.h"

#include <algorithm>
#include <string>
#include <vector>

// ---- helpers -------------------------------------------------------------

// true if two tensors live in the same Metal buffer
static bool ggml_metal_fusion_same_buffer(const ggml_tensor * a, const ggml_tensor * b) {
    if (!a || !b) {
        return false;
    }

    ggml_backend_buffer_t ba = a->view_src ? a->view_src->buffer : a->buffer;
    ggml_backend_buffer_t bb = b->view_src ? b->view_src->buffer : b->buffer;

    ggml_metal_buffer_t ca = (ggml_metal_buffer_t) ba->context;
    ggml_metal_buffer_t cb = (ggml_metal_buffer_t) bb->context;

    return ggml_metal_buffer_get_id(ca, a).metal == ggml_metal_buffer_get_id(cb, b).metal;
}

// ---- pattern checks ------------------------------------------------------

// NORM/RMS_NORM + MUL + ADD: the weight/bias of each fused step must match the norm input
// width, be contiguous rows, and the fused outputs must stay F32
static bool ggml_metal_fusion_check_norm(
        const ggml_metal_fusion      * fusion,
        const ggml_cgraph *,
        const int *,
        const ggml_tensor * const    * nodes,
              ggml_metal_fusion_mode   mode) {
    GGML_UNUSED(mode);

    GGML_ASSERT(fusion->n_ops >= 2);

    for (int j = 1; j < fusion->n_ops; j++) {
        // the fused MUL/ADD must read the previous node as src0
        if (nodes[j]->src[0] != nodes[j - 1]) {
            return false;
        }

        // the weight/bias must have the same row width as the norm input
        if (nodes[j]->src[1]->ne[0] != nodes[0]->ne[0]) {
            return false;
        }

        if (!ggml_is_contiguous_rows(nodes[j]->src[1])) {
            return false;
        }

        if (nodes[j]->type != GGML_TYPE_F32) {
            return false;
        }
    }

    return true;
}

// ADD x N: each ADD reads the previous ADD as src0, and all addends must share layout
// (and, in FULL mode, live in the same Metal buffer)
static bool ggml_metal_fusion_check_add_chain(
        const ggml_metal_fusion      * fusion,
        const ggml_cgraph *,
        const int *,
        const ggml_tensor * const    * nodes,
              ggml_metal_fusion_mode   mode) {
    GGML_ASSERT(fusion->n_ops >= 2);

    for (int j = 1; j < fusion->n_ops; j++) {
        if (nodes[j]->src[0] != nodes[j - 1]) {
            return false;
        }

        if (!ggml_are_same_layout(nodes[j]->src[1], nodes[j - 1]->src[1])) {
            return false;
        }

        if (mode == GGML_METAL_FUSION_FULL) {
            if (!ggml_metal_fusion_same_buffer(nodes[j]->src[1], nodes[0]->src[1])) {
                return false;
            }
        }
    }

    return true;
}

static bool ggml_metal_overlaps_gdn_tail(
        const ggml_tensor * tensor,
        const ggml_tensor * gdn,
        size_t tail_off,
        size_t tail_size) {
    size_t tensor_off;
    if (tensor == gdn) {
        tensor_off = 0;
    } else if (tensor->view_src == gdn) {
        tensor_off = tensor->view_offs;
    } else {
        return false;
    }

    const size_t tensor_size = ggml_nbytes(tensor);
    return tensor_off < tail_off + tail_size && tail_off < tensor_off + tensor_size;
}

// The snapshot tail stays unwritten, so all other consumers must use only attention scores.
static bool ggml_metal_fusion_check_gdn_cache(
        const ggml_metal_fusion *,
        const ggml_cgraph * gf,
        const int *,
        const ggml_tensor * const * nodes,
        ggml_metal_fusion_mode mode) {
    const ggml_tensor * gdn = nodes[0];
    const ggml_tensor * cpy = nodes[1];
    if (gdn->type != GGML_TYPE_F32 || (gdn->flags & GGML_TENSOR_FLAG_OUTPUT) || (cpy->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return false;
    }

    const ggml_tensor * src_v     = gdn->src[2];
    const int64_t       S_v       = src_v->ne[0];
    const int64_t       H         = src_v->ne[1];
    const int64_t       n_tokens  = src_v->ne[2];
    const int64_t       n_seqs    = src_v->ne[3];
    const int64_t       D         = S_v*S_v*H;
    const int64_t       K         = ggml_get_op_params_i32(gdn, 0);
    const int64_t       n_written = std::min<int64_t>(n_tokens, K);
    const size_t        tail_off  = ggml_row_size(GGML_TYPE_F32, S_v*H*n_tokens*n_seqs);

    const ggml_tensor * src = cpy->src[0];
    const ggml_tensor * dst = cpy->src[1];
    const int64_t state_elements = D*n_seqs*n_written;
    if (src->op != GGML_OP_VIEW || src->view_src != gdn || src->view_offs != tail_off ||
        src->type != GGML_TYPE_F32 || !ggml_is_contiguous(src) || ggml_nelements(src) != state_elements) {
        return false;
    }

    if (dst->op != GGML_OP_VIEW || dst->type != GGML_TYPE_F32 || (mode == GGML_METAL_FUSION_FULL && dst->data == nullptr) ||
        dst->ne[0] != D || dst->ne[1] != n_seqs || dst->ne[2] != n_written || dst->ne[3] != 1 ||
        ggml_nelements(dst) != state_elements ||
        dst->nb[0] != ggml_type_size(GGML_TYPE_F32) || dst->nb[1] != ggml_row_size(GGML_TYPE_F32, D) ||
        dst->nb[2] < ggml_row_size(GGML_TYPE_F32, D*n_seqs) ||
        dst->nb[2] % sizeof(float) != 0 || dst->view_offs % sizeof(float) != 0) {
        return false;
    }

    const size_t tail_size = ggml_nbytes(src);
    if (tail_off > ggml_nbytes(gdn) || tail_size > ggml_nbytes(gdn) - tail_off) {
        return false;
    }
    for (int i = 0; i < gf->n_nodes; ++i) {
        const ggml_tensor * node = gf->nodes[i];
        if (node == gdn || node == cpy) {
            continue;
        }
        if ((node->flags & GGML_TENSOR_FLAG_OUTPUT) &&
            ggml_metal_overlaps_gdn_tail(node, gdn, tail_off, tail_size)) {
            return false;
        }
        if (ggml_op_is_empty(node->op) || (node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }
        for (int is = 0; is < GGML_MAX_SRC; ++is) {
            if (node->src[is] && ggml_metal_overlaps_gdn_tail(node->src[is], gdn, tail_off, tail_size)) {
                return false;
            }
        }
    }

    return true;
}

// MUL + SIN + SQR + MUL + ADD (snake activation)
static bool ggml_metal_fusion_check_snake(
        const ggml_metal_fusion      * fusion,
        const ggml_cgraph *,
        const int *,
        const ggml_tensor * const    * nodes,
              ggml_metal_fusion_mode   mode) {
    GGML_UNUSED(fusion);
    GGML_UNUSED(mode);

    const ggml_tensor * mul0     = nodes[0];
    const ggml_tensor * sin_node = nodes[1];
    const ggml_tensor * sqr      = nodes[2];
    const ggml_tensor * mul1     = nodes[3];
    const ggml_tensor * add      = nodes[4];

    // x carries the full activation shape, a is the broadcast operand
    const ggml_tensor * x = ggml_are_same_shape(mul0, mul0->src[0]) ? mul0->src[0] : mul0->src[1];
    const ggml_tensor * a = (x == mul0->src[0]) ? mul0->src[1] : mul0->src[0];

    // mul1 reads sqr and inv_b in either operand order
    const ggml_tensor * inv_b = (mul1->src[0] == sqr) ? mul1->src[1] : mul1->src[0];

    // closure check: the trailing add reads the same x as the leading mul
    const ggml_tensor * x_in_add = (add->src[0] == mul1) ? add->src[1] : add->src[0];

    // x is in the supported whitelist and every chain intermediate shares x's type.
    // a and inv_b bind as device const float * in the kernel, so they stay F32.
    const bool types_ok =
        (x->type == GGML_TYPE_F32 || x->type == GGML_TYPE_F16 || x->type == GGML_TYPE_BF16) &&
        (a->type    == GGML_TYPE_F32) && (inv_b->type    == GGML_TYPE_F32) &&
        (mul0->type == x->type)       && (sin_node->type == x->type) &&
        (sqr->type  == x->type)       && (mul1->type     == x->type) &&
        (add->type  == x->type);

    // a / inv_b collapse to [1, C, 1, 1], x and add stay 2D
    const bool shape_ok = ggml_are_same_shape(a, inv_b) && a->ne[0] == 1 && a->ne[1] == x->ne[1];
    const bool dim_ok =
        (x->ne[2]     == 1) && (x->ne[3]     == 1) &&
        (add->ne[2]   == 1) && (add->ne[3]   == 1) &&
        (a->ne[2]     == 1) && (a->ne[3]     == 1) &&
        (inv_b->ne[2] == 1) && (inv_b->ne[3] == 1);

    // kernel reads x[idx] and a[c] / inv_b[c] linearly, so every operand is contiguous
    const bool contig_ok =
        ggml_is_contiguous(x) && ggml_is_contiguous(add) &&
        ggml_is_contiguous(a) && ggml_is_contiguous(inv_b);

    return types_ok && shape_ok && dim_ok && contig_ok && x_in_add == x;
}

static bool ggml_metal_tensors_overlap(const ggml_tensor * a, const ggml_tensor * b) {
    const ggml_metal_buffer_id ba = ggml_metal_buffer_get_id((ggml_metal_buffer_t) (a->view_src ? a->view_src->buffer : a->buffer)->context, a);
    const ggml_metal_buffer_id bb = ggml_metal_buffer_get_id((ggml_metal_buffer_t) (b->view_src ? b->view_src->buffer : b->buffer)->context, b);
    if (ba.metal != bb.metal) {
        return false;
    }
    return ba.offs < bb.offs + ggml_nbytes(b) && bb.offs < ba.offs + ggml_nbytes(a);
}

static bool ggml_metal_moe_combine_view_matches(
        const ggml_tensor * view,
        const ggml_tensor * mul,
        int i_expert,
        int64_t n_embd,
        int64_t n_tokens) {
    return view && view->op == GGML_OP_VIEW && view->view_src == mul &&
        view->view_offs == (size_t) i_expert*mul->nb[1] &&
        view->type == GGML_TYPE_F32 && view->ne[0] == n_embd && view->ne[1] == n_tokens &&
        view->ne[2] == 1 && view->ne[3] == 1 && view->nb[0] == sizeof(float) &&
        view->nb[1] == mul->nb[2] && ggml_is_contiguous_rows(view);
}

static bool ggml_metal_fusion_check_moe_combine(
        const ggml_metal_fusion * fusion,
        const ggml_cgraph *,
        const int *,
        const ggml_tensor * const * nodes,
        ggml_metal_fusion_mode mode) {
    const ggml_tensor * mul = nodes[0];
    if (mul->op != GGML_OP_MUL || mul->type != GGML_TYPE_F32) {
        return false;
    }

    const ggml_tensor * experts = mul->src[0];
    const ggml_tensor * weights = mul->src[1];
    if (!experts || !weights || experts->type != GGML_TYPE_F32 || weights->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(experts) || !ggml_is_contiguous(weights) || !ggml_are_same_shape(experts, mul)) {
        return false;
    }

    const int64_t n_embd = experts->ne[0];
    const int64_t n_expert = experts->ne[1];
    const int64_t n_tokens = experts->ne[2];
    if (n_expert < 2 || n_expert > 15 || experts->ne[3] != 1 ||
        weights->ne[0] != 1 || weights->ne[1] != n_expert || weights->ne[2] != n_tokens || weights->ne[3] != 1 ||
        mul->ne[0] != n_embd || mul->ne[1] != n_expert || mul->ne[2] != n_tokens || mul->ne[3] != 1 ||
        fusion->n_ops != n_expert) {
        return false;
    }

    const ggml_tensor * previous = nullptr;
    for (int i = 1; i < n_expert; ++i) {
        const ggml_tensor * add = nodes[i];
        if (add->op != GGML_OP_ADD || add->type != GGML_TYPE_F32 ||
            add->ne[0] != n_embd || add->ne[1] != n_tokens || add->ne[2] != 1 || add->ne[3] != 1) {
            return false;
        }

        if (i == 1) {
            if (!ggml_metal_moe_combine_view_matches(add->src[0], mul, 0, n_embd, n_tokens) ||
                !ggml_metal_moe_combine_view_matches(add->src[1], mul, 1, n_embd, n_tokens)) {
                return false;
            }
        } else {
            const ggml_tensor * view = add->src[0] == previous ? add->src[1] : add->src[0];
            if ((add->src[0] != previous && add->src[1] != previous) ||
                !ggml_metal_moe_combine_view_matches(view, mul, i, n_embd, n_tokens)) {
                return false;
            }
        }
        previous = add;
    }

    const ggml_tensor * dst = nodes[n_expert - 1];
    if (!ggml_is_contiguous(dst) || (mode == GGML_METAL_FUSION_FULL && (ggml_metal_tensors_overlap(experts, dst) || ggml_metal_tensors_overlap(weights, dst)))) {
        return false;
    }

    return true;
}

static bool ggml_metal_fusion_check_moe_weights(
        const ggml_metal_fusion *,
        const ggml_cgraph *,
        const int *,
        const ggml_tensor * const * nodes,
        ggml_metal_fusion_mode) {
    const ggml_tensor * get_rows = nodes[0];
    const ggml_tensor * sum      = nodes[1];
    const ggml_tensor * clamp_op = nodes[2];
    const ggml_tensor * div      = nodes[3];
    const ggml_tensor * scale_op = nodes[4];
    const ggml_tensor * probs    = get_rows->src[0];
    const ggml_tensor * ids      = get_rows->src[1];
    const ggml_tensor * weights  = sum->src[0];
    const ggml_tensor * norm     = scale_op->src[0];

    const int64_t n_expert      = probs->ne[1];
    const int64_t n_expert_used = ids->ne[0];
    const int64_t n_tokens      = ids->ne[1];

    const bool can_fuse =
        get_rows->op == GGML_OP_GET_ROWS && sum->op == GGML_OP_SUM_ROWS && clamp_op->op == GGML_OP_CLAMP &&
        div->op == GGML_OP_DIV && scale_op->op == GGML_OP_SCALE &&
        probs->type == GGML_TYPE_F32 && ids->type == GGML_TYPE_I32 && get_rows->type == GGML_TYPE_F32 &&
        sum->type == GGML_TYPE_F32 && clamp_op->type == GGML_TYPE_F32 && div->type == GGML_TYPE_F32 && scale_op->type == GGML_TYPE_F32 &&
        weights->op == GGML_OP_RESHAPE && weights->src[0] == get_rows && sum->src[0] == weights && clamp_op->src[0] == sum &&
        div->src[0] == weights && div->src[1] == clamp_op && norm->op == GGML_OP_RESHAPE && norm->src[0] == div && scale_op->src[0] == norm &&
        probs->ne[0] == 1 && probs->ne[2] == n_tokens && probs->ne[3] == 1 &&
        n_expert > 0 && n_expert_used > 0 && n_expert_used <= 32 && n_expert_used <= n_expert &&
        ids->ne[2] == 1 && ids->ne[3] == 1 && get_rows->ne[0] == 1 && get_rows->ne[1] == n_expert_used &&
        get_rows->ne[2] == n_tokens && get_rows->ne[3] == 1 && ggml_are_same_shape(get_rows, scale_op) &&
        ggml_is_contiguous(probs) && ggml_is_contiguous(ids) && ggml_is_contiguous(scale_op);
    return can_fuse;
}

static bool ggml_metal_fusion_check_unary(
        const ggml_metal_fusion * fusion,
        const ggml_cgraph *,
        const int *,
        const ggml_tensor * const * nodes,
        ggml_metal_fusion_mode) {
    const ggml_tensor * op = nodes[0];
    const ggml_tensor * dst = nodes[1];
    if (dst->src[0] != op || op->src[0]->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (fusion->id == GGML_METAL_FUSION_SCALE_SILU) {
        return ggml_get_unary_op(dst) == GGML_UNARY_OP_SILU && ggml_is_contiguous_rows(op);
    }
    const ggml_unary_op unary = fusion->id == GGML_METAL_FUSION_SIGMOID_SCALE ? GGML_UNARY_OP_SIGMOID : GGML_UNARY_OP_SOFTPLUS;
    return ggml_get_unary_op(op) == unary && ggml_are_same_layout(op, dst);
}

static bool ggml_metal_fusion_check_norm_scale(
        const ggml_metal_fusion *,
        const ggml_cgraph *,
        const int *,
        const ggml_tensor * const * nodes,
        ggml_metal_fusion_mode) {
    return nodes[1]->src[0] == nodes[0] && nodes[1]->type == GGML_TYPE_F32 &&
        ggml_are_same_layout(nodes[0], nodes[1]) && ggml_get_op_params_f32(nodes[1], 1) == 0.0f;
}

static bool ggml_metal_fusion_check_rms_norm_rope(
        const ggml_metal_fusion * fusion,
        const ggml_cgraph *,
        const int *,
        const ggml_tensor * const * nodes,
        ggml_metal_fusion_mode) {
    const ggml_tensor * rms = nodes[0];
    const ggml_tensor * rope = nodes[1];
    const int mode   = ggml_get_op_params_i32(rope, 2);
    const int n_dims = ggml_get_op_params_i32(rope, 1);
    const int n_offs = ggml_get_op_params_i32(rope, 15);
    if (rope->src[0] != rms || rms->src[0]->type != GGML_TYPE_F32 || rms->type != GGML_TYPE_F32 ||
        rope->type != GGML_TYPE_F32 || rms->ne[0] != 512 || mode != GGML_ROPE_TYPE_NORMAL ||
        n_dims % 4 != 0 || n_offs % 4 != 0 || !ggml_are_same_layout(rms->src[0], rms) || !ggml_are_same_layout(rms, rope)) {
        return false;
    }
    if (fusion->n_ops == 3) {
        const ggml_tensor * cast = nodes[2];
        return cast->src[0] == rope && cast->src[1] == cast && cast->type == GGML_TYPE_F16 &&
            ggml_are_same_shape(rope, cast) && ggml_is_contiguous(cast);
    }
    return true;
}

static bool ggml_metal_fusion_check_dsv4_hc_affine(
        const ggml_metal_fusion *,
        const ggml_cgraph *,
        const int *,
        const ggml_tensor * const * nodes,
        ggml_metal_fusion_mode) {
    const ggml_tensor * mul      = nodes[0];
    const ggml_tensor * add      = nodes[1];
    const ggml_tensor * sigmoid  = nodes[2];
    const ggml_tensor * scale_op = nodes[3];
    const ggml_tensor * x        = mul->src[0];
    const ggml_tensor * scale    = mul->src[1];
    const ggml_tensor * base     = add->src[1];

    const bool can_fuse =
        add->src[0] == mul && sigmoid->src[0] == add && scale_op->src[0] == sigmoid &&
        ggml_get_unary_op(sigmoid) == GGML_UNARY_OP_SIGMOID &&
        x->type == GGML_TYPE_F32 && scale->type == GGML_TYPE_F32 && base->type == GGML_TYPE_F32 &&
        mul->type == GGML_TYPE_F32 && add->type == GGML_TYPE_F32 && sigmoid->type == GGML_TYPE_F32 && scale_op->type == GGML_TYPE_F32 &&
        x->ne[0] == 4 && x->ne[2] == 1 && x->ne[3] == 1 && ggml_nelements(scale) == 1 &&
        base->ne[0] == 4 && ggml_nrows(base) == 1 && ggml_are_same_shape(x, mul) && ggml_are_same_layout(mul, add) &&
        ggml_are_same_layout(add, sigmoid) && ggml_are_same_layout(sigmoid, scale_op) &&
        ggml_is_contiguous_rows(x) && ggml_is_contiguous(base) && ggml_is_contiguous_rows(scale_op);
    return can_fuse;
}

static bool ggml_metal_fusion_check_dsv4_hc_post_add(
        const ggml_metal_fusion *,
        const ggml_cgraph *,
        const int *,
        const ggml_tensor * const * nodes,
        ggml_metal_fusion_mode) {
    const ggml_tensor * add      = nodes[0];
    const ggml_tensor * hc_post  = nodes[1];
    const ggml_tensor * x        = add->src[0];
    const ggml_tensor * y        = add->src[1];
    const ggml_tensor * residual = hc_post->src[1];
    const ggml_tensor * post     = hc_post->src[2];
    const ggml_tensor * comb     = hc_post->src[3];

    const bool can_fuse =
        hc_post->src[0] == add &&
        x->type == GGML_TYPE_F32 && y->type == GGML_TYPE_F32 && add->type == GGML_TYPE_F32 &&
        residual->type == GGML_TYPE_F32 && post->type == GGML_TYPE_F32 && comb->type == GGML_TYPE_F32 && hc_post->type == GGML_TYPE_F32 &&
        ggml_are_same_layout(x, y) && ggml_are_same_shape(x, add) &&
        x->ne[2] == 1 && x->ne[3] == 1 && residual->ne[0] == x->ne[0] && residual->ne[1] == 4 && residual->ne[2] == x->ne[1] &&
        post->ne[0] == 4 && post->ne[1] == x->ne[1] && comb->ne[0] == 4 && comb->ne[1] == 4 && comb->ne[2] == x->ne[1] &&
        hc_post->ne[0] == x->ne[0] && hc_post->ne[1] == 4 && hc_post->ne[2] == x->ne[1] &&
        ggml_is_contiguous_rows(x) && ggml_is_contiguous_rows(hc_post);
    return can_fuse;
}

static bool ggml_metal_fusion_check_dsv4_hc_pre_norm(
        const ggml_metal_fusion *,
        const ggml_cgraph *,
        const int *,
        const ggml_tensor * const * nodes,
        ggml_metal_fusion_mode) {
    const ggml_tensor * op = nodes[0];
    const ggml_tensor * x = op->src[0];
    const ggml_tensor * weights = op->src[1];
    const ggml_tensor * norm = nodes[1];
    const ggml_tensor * mul  = nodes[2];
    const ggml_tensor * norm_weight = mul->src[1];

    const bool can_fuse =
        x->ne[0] == 4096 && ggml_is_contiguous_rows(x) && ggml_is_contiguous_rows(weights) &&
        norm->src[0] == op &&
        mul->src[0] == norm &&
        norm->type == GGML_TYPE_F32 &&
        mul->type == GGML_TYPE_F32 &&
        norm_weight->type == GGML_TYPE_F32 &&
        ggml_are_same_shape(op, norm) &&
        ggml_are_same_shape(norm, mul) &&
        ggml_nelements(norm_weight) == x->ne[0] &&
        ggml_is_contiguous_rows(norm_weight) &&
        ggml_is_contiguous_rows(mul);

    return can_fuse;
}

static bool ggml_metal_fusion_check_qwen4exp_hc_reduce(
        const ggml_metal_fusion *,
        const ggml_cgraph *,
        const int *,
        const ggml_tensor * const * nodes,
        ggml_metal_fusion_mode) {
    const ggml_tensor * sigmoid = nodes[0];
    const ggml_tensor * op = nodes[1];
    const ggml_tensor * gate = sigmoid->src[0];
    return ggml_get_unary_op(sigmoid) == GGML_UNARY_OP_SIGMOID && op->src[1] == sigmoid &&
        gate->type == GGML_TYPE_F32 && ggml_is_contiguous(gate) &&
        ggml_are_same_shape(gate, sigmoid) &&
        op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
        op->src[0]->ne[1] == 4 && op->src[0]->ne[3] == 1 && ggml_are_same_shape(op->src[0], op->src[1]) &&
        op->ne[0] == op->src[0]->ne[0] && op->ne[1] == op->src[0]->ne[2] && op->ne[2] == 1 && op->ne[3] == 1 &&
        ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]) && ggml_is_contiguous(op);
}

// ---- patterns ------------------------------------------------------------

static const ggml_op ops_norm_mul[]         = { GGML_OP_NORM, GGML_OP_MUL };
static const ggml_op ops_norm_mul_add[]     = { GGML_OP_NORM, GGML_OP_MUL, GGML_OP_ADD };
static const ggml_op ops_rms_norm_mul[]     = { GGML_OP_RMS_NORM, GGML_OP_MUL };
static const ggml_op ops_rms_norm_mul_add[] = { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ADD };

static const ggml_op ops_add_2[] = { GGML_OP_ADD, GGML_OP_ADD };
static const ggml_op ops_add_3[] = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
static const ggml_op ops_add_4[] = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
static const ggml_op ops_add_5[] = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
static const ggml_op ops_add_6[] = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
static const ggml_op ops_add_7[] = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
static const ggml_op ops_snake[] = { GGML_OP_MUL, GGML_OP_SIN, GGML_OP_SQR, GGML_OP_MUL, GGML_OP_ADD };

static const ggml_op ops_gdn_cache[] = { GGML_OP_GATED_DELTA_NET, GGML_OP_CPY };

static const ggml_op ops_scale_silu[] = { GGML_OP_SCALE, GGML_OP_UNARY };
static const ggml_op ops_sigmoid_scale[] = { GGML_OP_UNARY, GGML_OP_SCALE };
static const ggml_op ops_softplus_sqrt[] = { GGML_OP_UNARY, GGML_OP_SQRT };
static const ggml_op ops_norm_scale[] = { GGML_OP_NORM, GGML_OP_SCALE };
static const ggml_op ops_rms_norm_scale[] = { GGML_OP_RMS_NORM, GGML_OP_SCALE };
static const ggml_op ops_rms_norm_rope[] = { GGML_OP_RMS_NORM, GGML_OP_ROPE };
static const ggml_op ops_rms_norm_rope_cpy[] = { GGML_OP_RMS_NORM, GGML_OP_ROPE, GGML_OP_CPY };
static const ggml_op ops_moe_weights[] = { GGML_OP_GET_ROWS, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_SCALE };
static const ggml_op ops_moe_combine_2[] = { GGML_OP_MUL, GGML_OP_ADD };
static const ggml_op ops_moe_combine_3[] = { GGML_OP_MUL, GGML_OP_ADD, GGML_OP_ADD };
static const ggml_op ops_moe_combine_4[] = { GGML_OP_MUL, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
static const ggml_op ops_moe_combine_5[] = { GGML_OP_MUL, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
static const ggml_op ops_moe_combine_6[] = { GGML_OP_MUL, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
static const ggml_op ops_moe_combine_7[] = { GGML_OP_MUL, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
static const ggml_op ops_moe_combine_8[] = { GGML_OP_MUL, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
static const ggml_op ops_moe_combine_9[] = { GGML_OP_MUL, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
static const ggml_op ops_moe_combine_10[] = { GGML_OP_MUL, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
static const ggml_op ops_moe_combine_11[] = { GGML_OP_MUL, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
static const ggml_op ops_moe_combine_12[] = { GGML_OP_MUL, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
static const ggml_op ops_moe_combine_13[] = { GGML_OP_MUL, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
static const ggml_op ops_moe_combine_14[] = { GGML_OP_MUL, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
static const ggml_op ops_moe_combine_15[] = { GGML_OP_MUL, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };

static const ggml_op ops_dsv4_hc_affine[] = { GGML_OP_MUL, GGML_OP_ADD, GGML_OP_UNARY, GGML_OP_SCALE };
static const ggml_op ops_dsv4_hc_post_add[] = { GGML_OP_ADD, GGML_OP_DSV4_HC_POST };
static const ggml_op ops_dsv4_hc_pre_norm[] = { GGML_OP_DSV4_HC_PRE, GGML_OP_RMS_NORM, GGML_OP_MUL };

static const ggml_op ops_qwen4exp_hc_reduce[] = { GGML_OP_UNARY, GGML_OP_QWEN4EXP_HC_REDUCE };

static const ggml_metal_fusion ggml_metal_fusions[] = {
    { GGML_METAL_FUSION_QWEN4EXP_HC_REDUCE, ops_qwen4exp_hc_reduce, 2, GGML_METAL_FUSION_SUBGRAPH, ggml_metal_fusion_check_qwen4exp_hc_reduce },
    { GGML_METAL_FUSION_DSV4_HC_AFFINE, ops_dsv4_hc_affine, 4, GGML_METAL_FUSION_CHAIN, ggml_metal_fusion_check_dsv4_hc_affine },
    { GGML_METAL_FUSION_DSV4_HC_POST_ADD, ops_dsv4_hc_post_add, 2, GGML_METAL_FUSION_SUBGRAPH, ggml_metal_fusion_check_dsv4_hc_post_add },
    { GGML_METAL_FUSION_DSV4_HC_PRE_NORM, ops_dsv4_hc_pre_norm, 3, GGML_METAL_FUSION_CHAIN, ggml_metal_fusion_check_dsv4_hc_pre_norm },

    { GGML_METAL_FUSION_SCALE_SILU, ops_scale_silu, 2, GGML_METAL_FUSION_CHAIN, ggml_metal_fusion_check_unary },
    { GGML_METAL_FUSION_SIGMOID_SCALE, ops_sigmoid_scale, 2, GGML_METAL_FUSION_CHAIN, ggml_metal_fusion_check_unary },
    { GGML_METAL_FUSION_SOFTPLUS_SQRT, ops_softplus_sqrt, 2, GGML_METAL_FUSION_CHAIN, ggml_metal_fusion_check_unary },
    { GGML_METAL_FUSION_NORM_SCALE, ops_norm_scale, 2, GGML_METAL_FUSION_CHAIN, ggml_metal_fusion_check_norm_scale },
    { GGML_METAL_FUSION_NORM_SCALE, ops_rms_norm_scale, 2, GGML_METAL_FUSION_CHAIN, ggml_metal_fusion_check_norm_scale },
    { GGML_METAL_FUSION_RMS_NORM_ROPE, ops_rms_norm_rope, 2, GGML_METAL_FUSION_CHAIN, ggml_metal_fusion_check_rms_norm_rope },
    { GGML_METAL_FUSION_RMS_NORM_ROPE_CPY, ops_rms_norm_rope_cpy, 3, GGML_METAL_FUSION_CHAIN, ggml_metal_fusion_check_rms_norm_rope },
    { GGML_METAL_FUSION_MOE_WEIGHTS, ops_moe_weights, 5, GGML_METAL_FUSION_SUBGRAPH, ggml_metal_fusion_check_moe_weights },
    { GGML_METAL_FUSION_MOE_COMBINE, ops_moe_combine_2, 2, GGML_METAL_FUSION_SUBGRAPH, ggml_metal_fusion_check_moe_combine },
    { GGML_METAL_FUSION_MOE_COMBINE, ops_moe_combine_3, 3, GGML_METAL_FUSION_SUBGRAPH, ggml_metal_fusion_check_moe_combine },
    { GGML_METAL_FUSION_MOE_COMBINE, ops_moe_combine_4, 4, GGML_METAL_FUSION_SUBGRAPH, ggml_metal_fusion_check_moe_combine },
    { GGML_METAL_FUSION_MOE_COMBINE, ops_moe_combine_5, 5, GGML_METAL_FUSION_SUBGRAPH, ggml_metal_fusion_check_moe_combine },
    { GGML_METAL_FUSION_MOE_COMBINE, ops_moe_combine_6, 6, GGML_METAL_FUSION_SUBGRAPH, ggml_metal_fusion_check_moe_combine },
    { GGML_METAL_FUSION_MOE_COMBINE, ops_moe_combine_7, 7, GGML_METAL_FUSION_SUBGRAPH, ggml_metal_fusion_check_moe_combine },
    { GGML_METAL_FUSION_MOE_COMBINE, ops_moe_combine_8, 8, GGML_METAL_FUSION_SUBGRAPH, ggml_metal_fusion_check_moe_combine },
    { GGML_METAL_FUSION_MOE_COMBINE, ops_moe_combine_9, 9, GGML_METAL_FUSION_SUBGRAPH, ggml_metal_fusion_check_moe_combine },
    { GGML_METAL_FUSION_MOE_COMBINE, ops_moe_combine_10, 10, GGML_METAL_FUSION_SUBGRAPH, ggml_metal_fusion_check_moe_combine },
    { GGML_METAL_FUSION_MOE_COMBINE, ops_moe_combine_11, 11, GGML_METAL_FUSION_SUBGRAPH, ggml_metal_fusion_check_moe_combine },
    { GGML_METAL_FUSION_MOE_COMBINE, ops_moe_combine_12, 12, GGML_METAL_FUSION_SUBGRAPH, ggml_metal_fusion_check_moe_combine },
    { GGML_METAL_FUSION_MOE_COMBINE, ops_moe_combine_13, 13, GGML_METAL_FUSION_SUBGRAPH, ggml_metal_fusion_check_moe_combine },
    { GGML_METAL_FUSION_MOE_COMBINE, ops_moe_combine_14, 14, GGML_METAL_FUSION_SUBGRAPH, ggml_metal_fusion_check_moe_combine },
    { GGML_METAL_FUSION_MOE_COMBINE, ops_moe_combine_15, 15, GGML_METAL_FUSION_SUBGRAPH, ggml_metal_fusion_check_moe_combine },

    { GGML_METAL_FUSION_NORM_MUL,     ops_norm_mul,         2, GGML_METAL_FUSION_CHAIN, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_MUL_ADD, ops_norm_mul_add,     3, GGML_METAL_FUSION_CHAIN, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_MUL,     ops_rms_norm_mul,     2, GGML_METAL_FUSION_CHAIN, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_MUL_ADD, ops_rms_norm_mul_add, 3, GGML_METAL_FUSION_CHAIN, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_ADD_CHAIN,    ops_add_2,            2, GGML_METAL_FUSION_CHAIN, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_ADD_CHAIN,    ops_add_3,            3, GGML_METAL_FUSION_CHAIN, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_ADD_CHAIN,    ops_add_4,            4, GGML_METAL_FUSION_CHAIN, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_ADD_CHAIN,    ops_add_5,            5, GGML_METAL_FUSION_CHAIN, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_ADD_CHAIN,    ops_add_6,            6, GGML_METAL_FUSION_CHAIN, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_ADD_CHAIN,    ops_add_7,            7, GGML_METAL_FUSION_CHAIN, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_SNAKE,        ops_snake,            5, GGML_METAL_FUSION_CHAIN, ggml_metal_fusion_check_snake },
    { GGML_METAL_FUSION_GDN_CACHE,    ops_gdn_cache,        2, GGML_METAL_FUSION_UNSAFE, ggml_metal_fusion_check_gdn_cache },
};

const ggml_metal_fusion * ggml_metal_fusion_all(int * n) {
    *n = (int) sizeof(ggml_metal_fusions) / sizeof(ggml_metal_fusions[0]);

    return ggml_metal_fusions;
}

// ---- shared fusion info ---------------------------------------------------

static std::string ggml_metal_fusion_label(const ggml_metal_fusion * fusion) {
    GGML_ASSERT(fusion != nullptr);

    std::string label;
    for (int j = 0; j < fusion->n_ops; j++) {
        if (j > 0) {
            label += '+';
        }
        label += ggml_op_name(fusion->ops[j]);
    }
    return label;
}

struct ggml_metal_fusion_info {
    std::vector<std::string> labels;
    std::vector<uint64_t>    counts;
    bool enabled;
    bool stats;
    bool labels_set;
    int  debug;
};

struct ggml_metal_fusion_info * ggml_metal_fusion_info_init(bool enabled, int debug) {
    ggml_metal_fusion_info * finfo = new ggml_metal_fusion_info;
    finfo->enabled    = enabled;
    finfo->stats      = debug > 0;
    finfo->labels_set = false;
    finfo->debug      = debug;

    if (finfo->stats) {
        ggml_metal_fusion_info_labels_init(finfo);
    }

    return finfo;
}

void ggml_metal_fusion_info_free(struct ggml_metal_fusion_info * finfo) {
    delete finfo;
}

bool ggml_metal_fusion_info_enabled(const struct ggml_metal_fusion_info * finfo) {
    return finfo->enabled;
}

bool ggml_metal_fusion_info_stats(const struct ggml_metal_fusion_info * finfo) {
    return finfo->stats;
}

int ggml_metal_fusion_info_debug(const struct ggml_metal_fusion_info * finfo) {
    return finfo->debug;
}

int ggml_metal_fusion_info_n_fusions(const struct ggml_metal_fusion_info * finfo) {
    return (int) finfo->labels.size();
}

const char * ggml_metal_fusion_info_label(const struct ggml_metal_fusion_info * finfo, int idx) {
    GGML_ASSERT(idx >= 0 && idx < (int) finfo->labels.size());
    return finfo->labels[idx].c_str();
}

uint64_t ggml_metal_fusion_info_count(const struct ggml_metal_fusion_info * finfo, int idx) {
    GGML_ASSERT(idx >= 0 && idx < (int) finfo->counts.size());
    return finfo->counts[idx];
}

void ggml_metal_fusion_info_count_fusion(struct ggml_metal_fusion_info * finfo, const struct ggml_metal_fusion * fusion) {
    if (!finfo->stats || fusion == nullptr) {
        return;
    }

    int n = 0;
    const ggml_metal_fusion * all = ggml_metal_fusion_all(&n);

    int idx = -1;
    for (int i = 0; i < n; i++) {
        if (&all[i] == fusion) {
            idx = i;
            break;
        }
    }

    if (idx >= 0 && idx < (int) finfo->counts.size()) {
        finfo->counts[idx]++;
    }
}

void ggml_metal_fusion_info_set_enabled(struct ggml_metal_fusion_info * finfo, bool enabled) {
    finfo->enabled = enabled;
}

void ggml_metal_fusion_info_labels_init(struct ggml_metal_fusion_info * finfo) {
    if (finfo->labels_set) {
        return;
    }

    int n = 0;
    const ggml_metal_fusion * all = ggml_metal_fusion_all(&n);

    finfo->labels.clear();
    finfo->counts.assign(n, 0);
    finfo->labels.reserve(n);

    for (int i = 0; i < n; i++) {
        finfo->labels.emplace_back(ggml_metal_fusion_label(&all[i]));
    }

    finfo->labels_set = true;
}

void ggml_metal_fusion_info_stats_init(struct ggml_metal_fusion_info * finfo) {
    finfo->stats = true;
    ggml_metal_fusion_info_labels_init(finfo);
}

void ggml_metal_fusion_info_stats_reset(struct ggml_metal_fusion_info * finfo) {
    std::fill(finfo->counts.begin(), finfo->counts.end(), 0);
}

int ggml_metal_fusion_info_stats_get(const struct ggml_metal_fusion_info * finfo, const char ** labels, uint64_t * counts, int n) {
    const int n_fusions = (int) finfo->labels.size();

    if (labels == nullptr) {
        return n_fusions;
    }

    const int n_fill = std::min(n, n_fusions);
    for (int i = 0; i < n_fill; i++) {
        labels[i] = finfo->labels[i].c_str();
        if (counts != nullptr) {
            counts[i] = finfo->counts[i];
        }
    }

    return n_fill;
}

// ---- queries -------------------------------------------------------------

// find the longest pattern matching the node sequence starting at idx
// (idx is a position in node_idxs, which maps to graph node indices)
const ggml_metal_fusion * ggml_metal_fusion_next(
        const ggml_cgraph * gf,
        const int * node_idxs,
        int n_idxs,
        int idx,
        ggml_metal_fusion_mode mode,
        int * n_out) {
    int n = 0;
    const ggml_metal_fusion * all = ggml_metal_fusion_all(&n);

    const ggml_metal_fusion * res = nullptr;
    int best = 1;

    for (int i = 0; i < n; i++) {
        const ggml_metal_fusion * fusion = &all[i];

        // only look for a longer match than the current best
        if (fusion->n_ops <= best) {
            continue;
        }
        if (idx + fusion->n_ops > n_idxs) {
            continue;
        }

        const ggml_tensor * nodes[GGML_METAL_FUSION_MAX];

        // the op sequence must match exactly
        bool ok = true;
        for (int j = 0; j < fusion->n_ops; j++) {
            nodes[j] = gf->nodes[node_idxs[idx + j]];
            if (nodes[j]->op != fusion->ops[j]) {
                ok = false;
                break;
            }
        }
        if (!ok) {
            continue;
        }

        if (fusion->kind != GGML_METAL_FUSION_UNSAFE) {
            // common element-wise chain constraints: each node reads the previous one,
            // and all nodes have the same shape
            for (int j = 1; fusion->kind == GGML_METAL_FUSION_CHAIN && j < fusion->n_ops && ok; j++) {
                if (nodes[j]->src[0] != nodes[j - 1] && nodes[j]->src[1] != nodes[j - 1]) {
                    ok = false;
                    break;
                }
                if (!ggml_are_same_shape(nodes[j], nodes[j - 1])) {
                    ok = false;
                    break;
                }
            }
            if (!ok) {
                continue;
            }

            const int output = node_idxs[idx + fusion->n_ops - 1];
            if (fusion->kind == GGML_METAL_FUSION_SUBGRAPH) {
                const int first = node_idxs[idx];
                const int count = output - first + 1;
                if (count > GGML_METAL_FUSION_MAX) {
                    continue;
                }
                ggml_op ops[GGML_METAL_FUSION_MAX];
                for (int j = 0; j < count; ++j) {
                    ops[j] = gf->nodes[first + j]->op;
                }
                if (!ggml_can_fuse_subgraph(gf, first, count, ops, &output, 1)) {
                    continue;
                }
            } else if (!ggml_can_fuse_subgraph_ext(gf, node_idxs + idx, fusion->n_ops, fusion->ops, &output, 1)) {
                continue;
            }
        }

        // pattern-specific checks (the sole validator for unsafe patterns)
        if (fusion->check && !fusion->check(fusion, gf, node_idxs + idx, nodes, mode)) {
            continue;
        }

        best = fusion->n_ops;
        res = fusion;
    }

    *n_out = best;

    return res;
}

// optimize phase: maximum number of nodes starting at idx (a raw sequential graph index) that
// could be fused, chaining patterns back-to-back. matching runs on the same filtered (view
// transparent) node sequence that the compute phase uses, so the returned count is the raw index
// span from idx to the last matched node (intermediate views are packed along).
int ggml_metal_fusion_max(const ggml_cgraph * gf, int idx) {
    // an empty/view node cannot start a pattern - pack it alone
    if (ggml_op_is_empty(gf->nodes[idx]->op) || ggml_is_empty(gf->nodes[idx])) {
        return 1;
    }

    // collect the non-empty node indices starting at idx
    int idxs[GGML_METAL_FUSION_MAX];
    int n_idxs = 0;
    for (int i = idx; i < gf->n_nodes && n_idxs < GGML_METAL_FUSION_MAX; i++) {
        if (!ggml_op_is_empty(gf->nodes[i]->op) && !ggml_is_empty(gf->nodes[i])) {
            idxs[n_idxs++] = i;
        }
    }
    if (n_idxs == 0) {
        return 1;
    }

    int total = 0;
    int i_f = 0;

    while (i_f < n_idxs && total < GGML_METAL_FUSION_MAX) {
        int len = 1;
        const ggml_metal_fusion * fusion = ggml_metal_fusion_next(gf, idxs, n_idxs, i_f, GGML_METAL_FUSION_STRUCTURAL, &len);
        if (!fusion || total + len > GGML_METAL_FUSION_MAX) {
            break;
        }

        if (idxs[i_f + len - 1] - idx + 1 > GGML_METAL_FUSION_MAX) {
            break;
        }

        total += len;
        i_f += len;
    }

    if (i_f == 0) {
        return 1;
    }

    // map the matched non-empty nodes back to the raw index span (views are included)
    return std::min(GGML_METAL_FUSION_MAX, idxs[i_f - 1] - idx + 1);
}
