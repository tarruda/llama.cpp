#include "models.h"

static int deepseek4_vision_raw_tokens(int vit_h, int vit_w) {
    const int h = (vit_h + 2) / 3;
    const int w = (vit_w + 2) / 3;
    const int rows = h + h % 2;
    const int row_length = w + 1;
    const int tail = (rows / 2 * row_length % 2) * 2;
    return 3 + 1 + rows * row_length + tail + 1;
}

ggml_cgraph * clip_graph_deepseek4_vision::build() {
    const int vit_h = n_patches_y;
    const int vit_w = n_patches_x;

    ggml_tensor * pos_h = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_patches);
    ggml_set_name(pos_h, "pos_h");
    ggml_set_input(pos_h);

    ggml_tensor * pos_w = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_patches);
    ggml_set_name(pos_w, "pos_w");
    ggml_set_input(pos_w);

    auto add_pos = [&](ggml_tensor * cur, const clip_layer &) {
        return build_rope_2d(ctx0, cur, pos_h, pos_w, hparams.rope_theta, false);
    };

    ggml_tensor * cur = build_vit(
        build_inp(), n_patches, NORM_TYPE_RMS, FFN_SILU, nullptr, add_pos);
    cb(cur, "vit_out", -1);

    const int pad_w = (3 - vit_w % 3) % 3;
    const int pad_h = (3 - vit_h % 3) % 3;
    cur = ggml_reshape_3d(ctx0, cur, n_embd, vit_w, vit_h);
    if (pad_w || pad_h) {
        cur = ggml_pad(ctx0, cur, 0, pad_w, pad_h, 0);
    }
    cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 2, 0, 1, 3));

    ggml_tensor * kernel = ggml_view_3d(ctx0, cur, 3, 3, cur->ne[2], 0, 0, 0);
    cur = ggml_im2col(ctx0, kernel, cur, 3, 3, 0, 0, 1, 1, true, cur->type);
    cur = ggml_reshape_2d(ctx0, cur, cur->ne[0], cur->ne[1] * cur->ne[2]);
    cb(cur, "aligner_unfold", -1);

    cur = build_ffn(cur,
        model.mm_0_w, model.mm_0_b,
        nullptr, nullptr,
        model.mm_1_w, model.mm_1_b,
        FFN_GELU_ERF, -1);
    cb(cur, "aligner_out", -1);

    ggml_tensor * structural = ggml_reshape_2d(ctx0, model.image_pad, n_mmproj_embd, 1);
    structural = ggml_concat(ctx0, structural, ggml_reshape_2d(ctx0, model.mm_boi, n_mmproj_embd, 1), 1);
    structural = ggml_concat(ctx0, structural, ggml_reshape_2d(ctx0, model.image_newline, n_mmproj_embd, 1), 1);
    structural = ggml_concat(ctx0, structural, ggml_reshape_2d(ctx0, model.mm_eoi, n_mmproj_embd, 1), 1);
    cur = ggml_concat(ctx0, structural, cur, 1);

    const int n_output = deepseek4_vision_raw_tokens(vit_h, vit_w);
    ggml_tensor * indices = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_output);
    ggml_set_name(indices, "deepseek4_vision_indices");
    ggml_set_input(indices);
    cur = ggml_get_rows(ctx0, cur, indices);
    cb(cur, "image_block", -1);

    cur = ggml_cast(ctx0, ggml_cast(ctx0, cur, GGML_TYPE_BF16), GGML_TYPE_F32);
    cb(cur, "embeddings", -1);
    ggml_build_forward_expand(gf, cur);
    return gf;
}
