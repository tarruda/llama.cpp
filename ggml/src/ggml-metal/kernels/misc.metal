#include "common.h"
#include "dequantize.h"

kernel void kernel_argmax_f32(
        constant ggml_metal_kargs_argmax & args,
        device   const char * src0,
        device         char * dst,
        threadgroup    char * shmem [[threadgroup(0)]],
        uint  tgpig[[threadgroup_position_in_grid]],
        uint  tpitg[[thread_position_in_threadgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint    ntg[[threads_per_threadgroup]]) {
    device const float * x_row = (device const float *) ((device const char *) src0 + tgpig * args.nb01);

    float   lmax = -INFINITY;
    int32_t larg = -1;

    for (int i00 = tpitg; i00 < args.ne00; i00 += ntg) {
        if (x_row[i00] > lmax) {
            lmax = x_row[i00];
            larg = i00;
        }
    }

    // find the argmax value in the block
    float max_val = simd_max(lmax);
    int32_t arg_val = simd_max(select(-1, larg, lmax == max_val));

    device int32_t * dst_i32 = (device int32_t *) dst;

    threadgroup   float * shared_maxval = (threadgroup   float *) shmem;
    threadgroup int32_t * shared_argmax = (threadgroup int32_t *) shmem + N_SIMDWIDTH;

    if (ntg > N_SIMDWIDTH) {
        if (sgitg == 0) {
            shared_maxval[tiisg] = -INFINITY;
            shared_argmax[tiisg] = -1;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            shared_maxval[sgitg] = max_val;
            shared_argmax[sgitg] = arg_val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        max_val = shared_maxval[tiisg];
        arg_val = shared_argmax[tiisg];

        float max_val_reduced   = simd_max(max_val);
        int32_t arg_val_reduced = simd_max(select(-1, arg_val, max_val == max_val_reduced));

        dst_i32[tgpig] = arg_val_reduced;

        return;
    }

    dst_i32[tgpig] = arg_val;
}

kernel void kernel_diag_f32(
        constant ggml_metal_kargs_diag & args,
        device   const char * src0,
        device         char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]]) {
    constexpr short NW = N_SIMDWIDTH;

    const int32_t i3 = tgpig.z;
    const int32_t i2 = tgpig.y;
    const int32_t i1 = tgpig.x;

    device const float * src0_ptr = (device const float *)(src0 +                i2*args.nb02 + i3*args.nb03);
    device       float * dst_ptr  = (device       float *)(dst  + i1*args.nb01 + i2*args.nb2  + i3*args.nb3);

    for (int i0 = tiitg; i0 < args.ne0; i0 += NW) {
        dst_ptr[i0] = i0 == i1 ? src0_ptr[i0] : 0.0f;
    }
}

kernel void kernel_roll_f32(
    constant ggml_metal_kargs_roll & args,
    device  const char * src0,
    device        char * dst,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]]) {

    const int64_t i3 = tgpig.z;
    const int64_t i2 = tgpig.y;
    const int64_t i1 = tgpig.x;

    device const float * src0_ptr = (device const float *) src0;
    device       float * dst_ptr  = (device       float *) dst;

    for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {
        // apply shifts and wrap around
        int64_t i00 = i0 - args.s0;
        int64_t i01 = i1 - args.s1;
        int64_t i02 = i2 - args.s2;
        int64_t i03 = i3 - args.s3;

        if (i00 < 0) { i00 += args.ne00; } else if (i00 >= args.ne00) { i00 -= args.ne00; }
        if (i01 < 0) { i01 += args.ne01; } else if (i01 >= args.ne01) { i01 -= args.ne01; }
        if (i02 < 0) { i02 += args.ne02; } else if (i02 >= args.ne02) { i02 -= args.ne02; }
        if (i03 < 0) { i03 += args.ne03; } else if (i03 >= args.ne03) { i03 -= args.ne03; }

        int64_t src_idx = i03*args.ne02*args.ne01*args.ne00 + i02*args.ne01*args.ne00 + i01*args.ne00 + i00;
        int64_t dst_idx = i3 *args.ne2 *args.ne1 *args.ne0  + i2 *args.ne1 *args.ne0  + i1 *args.ne0  + i0;

        dst_ptr[dst_idx] = src0_ptr[src_idx];
    }
}

template <typename T>
kernel void kernel_pad_impl(
    constant ggml_metal_kargs_pad & args,
    device  const char * src0,
    device        char * dst,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]]) {
    const int32_t i3 = tgpig.z;
    const int32_t i2 = tgpig.y;
    const int32_t k0 = tgpig.x/args.ne1;
    const int32_t i1 = tgpig.x - k0*args.ne1;

    const int32_t i03 = i3;
    const int32_t i02 = i2;
    const int32_t i01 = i1;

    device const T * src0_ptr = (device const T *) (src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01);
    device       T * dst_ptr  = (device       T *) (dst  +  i3*args.nb3  +  i2*args.nb2  +  i1*args.nb1);

    for (int32_t l0 = 0; l0 < 1024; l0 += ntg.x) {
        const int32_t i0 = k0*1024 + tpitg.x + l0;
        if (i0 >= args.ne0) {
            break;
        }

        if (i0 < args.ne00 && i1 < args.ne01 && i2 < args.ne02 && i3 < args.ne03) {
            dst_ptr[i0] = src0_ptr[i0];
        } else {
            dst_ptr[i0] = 0.0f;
        }
    }
}

typedef decltype(kernel_pad_impl<float>) kernel_pad_t;

template [[host_name("kernel_pad_f32")]]   kernel kernel_pad_t kernel_pad_impl<float>;
template [[host_name("kernel_pad_f32_4")]] kernel kernel_pad_t kernel_pad_impl<float4>;

// TODO: this is slow - optimize
kernel void kernel_pad_reflect_1d_f32(
    constant   ggml_metal_kargs_pad_reflect_1d & args,
    device  const char * src0,
    device        char * dst,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3  tgpg[[threadgroups_per_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]]) {

    const int64_t i3 = tgpig.z;
    const int64_t i2 = tgpig.y;
    const int64_t i1 = tgpig.x;

    const int64_t i03 = i3;
    const int64_t i02 = i2;
    const int64_t i01 = i1;

    device const float * src0_ptr = (device const float *) (src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01);
    device       float * dst_ptr  = (device       float *) (dst  +  i3*args.nb3  +  i2*args.nb2  +  i1*args.nb1);

    if (i1 < args.ne01 && i2 < args.ne02 && i3 < args.ne03) {
        for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {
            if (i0 < args.p0) {
                dst_ptr[i0] = src0_ptr[args.p0 - i0];
            } else if (i0 < args.ne0 - args.p1) {
                dst_ptr[i0] = src0_ptr[i0 - args.p0];
            } else {
                dst_ptr[i0] = src0_ptr[(args.ne0 - args.p1 - args.p0) - (args.p1 + 1 - (args.ne0 - i0)) - 1];
            }
        }
    }
}

kernel void kernel_arange_f32(
    constant   ggml_metal_kargs_arange & args,
    device        char * dst,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]]) {

    device float * dst_ptr = (device float *) dst;

    for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {
        dst_ptr[i0] = args.start + args.step * i0;
    }
}

kernel void kernel_timestep_embedding_f32(
    constant  ggml_metal_kargs_timestep_embedding & args,
    device  const char * src0,
    device        char * dst,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]]) {

    int i = tgpig.x;
    device float * embed_data = (device float *)(dst + i*args.nb1);

    int half_ = args.dim / 2;
    for (int j = tpitg.x; j < half_; j += ntg.x) {
        float timestep = ((device float *)src0)[i];
        float freq = (float)exp(-log((float)args.max_period) * j / half_);
        float arg = timestep * freq;
        embed_data[j        ] = cos(arg);
        embed_data[j + half_] = sin(arg);
    }

    if (args.dim % 2 != 0 && tpitg.x == 0) {
        embed_data[2 * half_] = 0.f;
    }
}

kernel void kernel_opt_step_adamw_f32(
        constant    ggml_metal_kargs_opt_step_adamw & args,
        device       float * x,
        device const float * g,
        device       float * g_m,
        device       float * g_v,
        device const float * pars,
        uint        gid[[thread_position_in_grid]]) {

    if (gid >= args.np) {
        return;
    }

    const float alpha  = pars[0];
    const float beta1  = pars[1];
    const float beta2  = pars[2];
    const float eps    = pars[3];
    const float wd     = pars[4];
    const float beta1h = pars[5];
    const float beta2h = pars[6];

    const float gi = g[gid];
    const float gmi = g_m[gid] * beta1 +      gi * (1.0f - beta1);
    const float gvi = g_v[gid] * beta2 + gi * gi * (1.0f - beta2);

    g_m[gid] = gmi;
    g_v[gid] = gvi;

    const float mh =      gmi * beta1h;
    const float vh = sqrt(gvi * beta2h) + eps;

    x[gid] = x[gid] * (1.0f - alpha * wd) - alpha * mh / vh;
}

kernel void kernel_opt_step_sgd_f32(
        constant    ggml_metal_kargs_opt_step_sgd & args,
        device       float * x,
        device const float * g,
        device const float * pars,
        uint        gid[[thread_position_in_grid]]) {

    if (gid >= args.np) {
        return;
    }

    x[gid] = x[gid] * (1.0f - pars[0] * pars[1]) - pars[0] * g[gid];
}

template<typename T>
kernel void kernel_memset(
        constant ggml_metal_kargs_memset & args,
        device T * dst,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = args.val;
}

typedef decltype(kernel_memset<int64_t>) kernel_memset_t;

template [[host_name("kernel_memset_i64")]] kernel kernel_memset_t kernel_memset<int64_t>;

constant short FC_count_equal_nsg [[function_constant(FC_COUNT_EQUAL + 0)]];

template<typename T>
kernel void kernel_count_equal(
        constant ggml_metal_kargs_count_equal & args,
        device   const char * src0,
        device   const char * src1,
        device   atomic_int * dst,
        threadgroup int32_t * shmem_i32 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const short NSG = FC_count_equal_nsg;

    const int i3 = tgpig.z;
    const int i2 = tgpig.y;
    const int i1 = tgpig.x;

    if (i3 >= args.ne03 || i2 >= args.ne02 || i1 >= args.ne01) {
        return;
    }

    int sum = 0;

    device const char * base0 = src0 + i1*args.nb01 + i2*args.nb02 + i3*args.nb03;
    device const char * base1 = src1 + i1*args.nb11 + i2*args.nb12 + i3*args.nb13;

    for (int64_t i0 = tpitg.x; i0 < args.ne00; i0 += ntg.x) {
        const T v0 = *(device const T *)(base0 + i0*args.nb00);
        const T v1 = *(device const T *)(base1 + i0*args.nb10);
        sum += (v0 == v1);
    }

    sum = simd_sum(sum);

    if (tiisg == 0) {
        shmem_i32[sgitg] = sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg == 0) {
        float v = 0.0f;
        if (tpitg.x < NSG) {
            v = shmem_i32[tpitg.x];
        }

        float total = simd_sum(v);
        if (tpitg.x == 0) {
            atomic_fetch_add_explicit(dst, (int32_t) total, memory_order_relaxed);
        }
    }
}

typedef decltype(kernel_count_equal<int32_t>) kernel_count_equal_t;

template [[host_name("kernel_count_equal_i32")]] kernel kernel_count_equal_t kernel_count_equal<int32_t>;

template <typename T>
kernel void kernel_snake(
        constant ggml_metal_kargs_snake & args,
        device const T     * x,
        device const float * a,
        device const float * inv_b,
        device       T     * dst,
        uint         tgpig [[threadgroup_position_in_grid]],
        uint         tpitg [[thread_position_in_threadgroup]],
        uint         ntg   [[threads_per_threadgroup]]) {

    const int idx = tgpig * ntg + tpitg;
    if (idx >= args.T * args.C) {
        return;
    }

    const int   c  = idx / args.T;  // x is [T, C], a / inv_b collapse to [1, C]
    const float xi = float(x[idx]);
    const float si = sin(a[c] * xi);
    dst[idx] = T(xi + si * si * inv_b[c]);
}

template [[host_name("kernel_snake_f32")]]  kernel void kernel_snake<float>(constant ggml_metal_kargs_snake &, device const float *, device const float *, device const float *, device float *, uint, uint, uint);
template [[host_name("kernel_snake_f16")]]  kernel void kernel_snake<half>(constant ggml_metal_kargs_snake &, device const half *, device const float *, device const float *, device half *, uint, uint, uint);
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_snake_bf16")]] kernel void kernel_snake<bfloat>(constant ggml_metal_kargs_snake &, device const bfloat *, device const float *, device const float *, device bfloat *, uint, uint, uint);
#endif

template<int N>
kernel void kernel_fwht_f32(
        constant ggml_metal_kargs_fwht & args,
        device const float * src,
        device float * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort3  ntg[[threads_per_threadgroup]]) {

    constexpr int NW = N_SIMDWIDTH;
    constexpr int NE = N / NW;

    const float scale = 1.0f / sqrt((float) N);

    const int sg_per_tg = ntg.x / NW;
    const int64_t r = tgpig.x * sg_per_tg + sgitg;
    if (r >= args.nrows) {
        return;
    }

    src += r * N;
    dst += r * N;

    const int lane = tiisg;

    float reg[NE];
    for (int i = 0; i < NE; i++) {
        reg[i] = src[i*NW + lane]*scale;
    }
    for (int i = 1; i < NW; i *= 2) {
        for (int j = 0; j < NE; j++) {
            const float val = reg[j];
            const float val2 = simd_shuffle_xor(val, i);
            reg[j] = (lane & i) == 0 ? val2 + val : val2 - val;
        }
    }

    for (int i = NW; i < N; i *= 2) {
        const int step = i / NW;
        for (int j = 0; j < NE; j += (2 * step)) {
            for (int k = 0; k < step; k++) {
                const float x = reg[j + k ];
                const float y = reg[j + k + step];
                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }

    for (int i = 0; i < NE; i++) {
        dst[i*NW + lane] = reg[i];
    }
}

typedef decltype(kernel_fwht_f32<64>) kernel_fwht_t;

template [[host_name("kernel_fwht_f32_64")]]  kernel kernel_fwht_t kernel_fwht_f32<64>;
template [[host_name("kernel_fwht_f32_128")]] kernel kernel_fwht_t kernel_fwht_f32<128>;
template [[host_name("kernel_fwht_f32_256")]] kernel kernel_fwht_t kernel_fwht_f32<256>;
template [[host_name("kernel_fwht_f32_512")]] kernel kernel_fwht_t kernel_fwht_f32<512>;

kernel void kernel_moe_combine_f32(
        constant ggml_metal_kargs_moe_combine & args,
        device const float * experts,
        device const float * weights,
        device       float * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]]) {
    const int it = tgpig.y;
    const int i0 = (int) tgpig.x*32 + tiisg;

    const float weight_lane = tiisg < args.n_expert ? weights[it*args.n_expert + tiisg] : 0.0f;
    float result = 0.0f;
    for (int ie = 0; ie < args.n_expert; ++ie) {
        const float weight = simd_shuffle(weight_lane, ie);
        if (i0 < args.n_embd) {
            const int idx = (it*args.n_expert + ie)*args.n_embd + i0;
            result = fma(experts[idx], weight, result);
        }
    }

    if (i0 < args.n_embd) {
        dst[it*args.n_embd + i0] = result;
    }
}

kernel void kernel_moe_weights_f32(
        constant ggml_metal_kargs_moe_weights & args,
        device const char * probs,
        device const char * ids,
        device       char * dst,
        uint    it[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]]) {
    if (it >= args.n_tokens) {
        return;
    }

    float weight = 0.0f;
    if (tiisg < args.n_expert_used) {
        const int32_t id = *(device const int32_t *) (ids + tiisg*args.nb_i0 + it*args.nb_i1);
        weight = *(device const float *) (probs + id*args.nb_p1 + it*args.nb_p2);
    }

    const float sum = clamp(simd_sum(weight), args.clamp_min, args.clamp_max);
    if (tiisg < args.n_expert_used) {
        *(device float *) (dst + tiisg*args.nb_d1 + it*args.nb_d2) = weight/sum*args.scale + args.bias;
    }
}

kernel void kernel_dsv4_hc_comb_f32(
        constant ggml_metal_kargs_dsv4_hc_comb & args,
        device const char * mixes,
        device const char * scale,
        device const char * base,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    constexpr ushort hc = 4;
    constexpr ushort comb_offset = 2*hc;

    const int it = tgpig.x*ntg.y + sgitg;
    if (it >= args.n_tokens) {
        return;
    }

    float scale_lane = 0.0f;
    if (tiisg == 0) {
        scale_lane = *(device const float *) (scale + 2*args.nb_s0);
    }
    const float scale_comb = simd_shuffle(scale_lane, 0);

    float v = 0.0f;
    if (tiisg < hc*hc) {
        v = *(device const float *) (mixes + (comb_offset + tiisg)*args.nb_m0 + it*args.nb_m1)*scale_comb
          + *(device const float *) (base   + (comb_offset + tiisg)*args.nb_b0);
    }

    // Softmax across destinations (the four contiguous lanes for each source).
    float vmax = max(v, simd_shuffle_xor(v, 1));
    vmax = max(vmax, simd_shuffle_xor(vmax, 2));
    v = exp(v - vmax);

    float sum = v + simd_shuffle_xor(v, 1);
    sum += simd_shuffle_xor(sum, 2);
    v = v/sum + args.eps;

    // Normalize columns: equal destination indices are four lanes apart.
    sum = v + simd_shuffle_xor(v, 4);
    sum += simd_shuffle_xor(sum, 8);
    v /= sum + args.eps;

    for (int i = 1; i < args.n_iter; ++i) {
        sum = v + simd_shuffle_xor(v, 1);
        sum += simd_shuffle_xor(sum, 2);
        v /= sum + args.eps;

        sum = v + simd_shuffle_xor(v, 4);
        sum += simd_shuffle_xor(sum, 8);
        v /= sum + args.eps;
    }

    if (tiisg < hc*hc) {
        const ushort idst = tiisg & 3;
        const ushort isrc = tiisg >> 2;
        *(device float *) (dst + idst*args.nb_d0 + isrc*args.nb_d1 + it*args.nb_d2) = v;
    }
}

kernel void kernel_dsv4_hc_pre_f32(
        constant ggml_metal_kargs_dsv4_hc_pre & args,
        device const char * x,
        device const char * weights,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    constexpr ushort hc = 4;

    const int it = tgpig.y;
    const int i0 = ((int) tgpig.x*ntg.y + sgitg)*32 + tiisg;

    float weight_lane = 0.0f;
    if (tiisg < hc) {
        weight_lane = *(device const float *) (weights + tiisg*args.nb_w0 + it*args.nb_w1);
    }

    float w[hc];
    FOR_UNROLL (ushort ih = 0; ih < hc; ++ih) {
        w[ih] = simd_shuffle(weight_lane, ih);
    }

    if (i0 >= args.n_embd) {
        return;
    }

    device const char * xb = x + i0*args.nb_x0 + it*args.nb_x2;
    float result = 0.0f;
    FOR_UNROLL (ushort ih = 0; ih < hc; ++ih) {
        result = fma(*(device const float *) (xb + ih*args.nb_x1), w[ih], result);
    }

    *(device float *) (dst + i0*args.nb_d0 + it*args.nb_d1) = result;
}

kernel void kernel_dsv4_hc_pre_norm_f32(
        constant ggml_metal_kargs_dsv4_hc_pre_norm & args,
        device const char * x,
        device const char * weights,
        device const char * norm,
        device       char * dst,
        threadgroup float * shmem_f32 [[threadgroup(0)]],
        uint    tgpig[[threadgroup_position_in_grid]],
        ushort  tpitg[[thread_position_in_threadgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  tiisg[[thread_index_in_simdgroup]]) {
    constexpr ushort hc = 4;

    float weight_lane = 0.0f;
    if (tiisg < hc) {
        weight_lane = *(device const float *) (weights + tiisg*args.nb_w0 + tgpig*args.nb_w1);
    }

    float4 result = 0.0f;
    device const char * xb = x + 4*tpitg*args.nb_x0 + tgpig*args.nb_x2;
    FOR_UNROLL (ushort ih = 0; ih < hc; ++ih) {
        const float weight = simd_shuffle(weight_lane, ih);
        result = fma(*(device const float4 *) (xb + ih*args.nb_x1), weight, result);
    }

    float sumf = simd_sum(dot(result, result));

    if (tiisg == 0) {
        shmem_f32[sgitg] = sumf;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    sumf = simd_sum(shmem_f32[tiisg]);
    const float scale = 1.0f/sqrt(sumf/args.n_embd + args.eps);

    const float4 norm_v = *(device const float4 *) (norm + 4*tpitg*args.nb_n0);
    *(device float4 *) (dst + 4*tpitg*args.nb_d0 + tgpig*args.nb_d1) = result*scale*norm_v;
}

template <bool add_x>
kernel void kernel_dsv4_hc_post_f32_impl(
        constant ggml_metal_kargs_dsv4_hc_post & args,
        device const char * x,
        device const char * y,
        device const char * residual,
        device const char * post,
        device const char * comb,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    constexpr ushort hc = 4;

    const int it = tgpig.y;
    const int i0 = ((int) tgpig.x*ntg.y + sgitg)*32 + tiisg;

    float coeff_lane = 0.0f;
    if (tiisg < hc) {
        coeff_lane = *(device const float *) (post + tiisg*args.nb_p0 + it*args.nb_p1);
    } else if (tiisg < hc + hc*hc) {
        const ushort idx  = tiisg - hc;
        const ushort idst = idx & 3;
        const ushort isrc = idx >> 2;
        coeff_lane = *(device const float *) (comb + idst*args.nb_c0 + isrc*args.nb_c1 + it*args.nb_c2);
    }

    float post_reg[hc];
    float comb_reg[hc][hc];
    FOR_UNROLL (ushort idst = 0; idst < hc; ++idst) {
        post_reg[idst] = simd_shuffle(coeff_lane, idst);
    }
    FOR_UNROLL (ushort isrc = 0; isrc < hc; ++isrc) {
        FOR_UNROLL (ushort idst = 0; idst < hc; ++idst) {
            comb_reg[isrc][idst] = simd_shuffle(coeff_lane, hc + idst + hc*isrc);
        }
    }

    if (i0 >= args.n_embd) {
        return;
    }

    float xv = *(device const float *) (x + i0*args.nb_x0 + it*args.nb_x1);
    if (add_x) {
        xv += *(device const float *) (y + i0*args.nb_x0 + it*args.nb_x1);
    }
    float result[hc];
    FOR_UNROLL (ushort idst = 0; idst < hc; ++idst) {
        result[idst] = xv*post_reg[idst];
    }

    device const char * rb = residual + i0*args.nb_r0 + it*args.nb_r2;
    FOR_UNROLL (ushort isrc = 0; isrc < hc; ++isrc) {
        const float rv = *(device const float *) (rb + isrc*args.nb_r1);
        FOR_UNROLL (ushort idst = 0; idst < hc; ++idst) {
            result[idst] = fma(rv, comb_reg[isrc][idst], result[idst]);
        }
    }

    FOR_UNROLL (ushort idst = 0; idst < hc; ++idst) {
        *(device float *) (dst + i0*args.nb_d0 + idst*args.nb_d1 + it*args.nb_d2) = result[idst];
    }
}

typedef decltype(kernel_dsv4_hc_post_f32_impl<false>) kernel_dsv4_hc_post_f32_t;

template [[host_name("kernel_dsv4_hc_post_f32")]]     kernel kernel_dsv4_hc_post_f32_t kernel_dsv4_hc_post_f32_impl<false>;
template [[host_name("kernel_dsv4_hc_post_add_f32")]] kernel kernel_dsv4_hc_post_f32_t kernel_dsv4_hc_post_f32_impl<true>;

kernel void kernel_dsv4_hc_affine_f32(
        constant ggml_metal_kargs_dsv4_hc_affine & args,
        device const char * x,
        device const float * scale,
        device const float4 * base,
        device       char * dst,
        uint it[[thread_position_in_grid]]) {
    if (it >= args.n_tokens) {
        return;
    }

    const float4 z = *(device const float4 *) (x + it*args.nb_x1) * scale[0] + base[0];
    const float4 result = 1.0f / (1.0f + exp(-z));
    *(device float4 *) (dst + it*args.nb_d1) = result*args.post_scale + args.post_bias;
}

kernel void kernel_dsv4_compress(
        constant ggml_metal_kargs_dsv4_compress & args,
        device const char * kv_state,
        device const char * score_state,
        device const char * read_idxs,
        device       char * dst,
        threadgroup int32_t * idxs [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiitg[[thread_index_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int n_read = (args.overlap ? 2 : 1)*args.ratio;
    const int ib = tgpig.y;

    for (int j = tiitg; j < n_read; j += ntg.x) {
        const bool cur_half = args.overlap && j >= args.ratio;
        const int jr = cur_half ? j - args.ratio : j;
        const int idx_pos = (cur_half ? args.ratio*args.n_blocks : 0) + ib*args.ratio + jr;
        idxs[j] = *(device const int32_t *) (read_idxs + idx_pos*args.nb_i0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (!args.overlap && args.ratio == 128) {
        const int lane = tiitg & 31;
        const int nsg = ntg.x/32;
        const int i0 = tgpig.x*(2*nsg) + (tiitg >> 5)*2 + (lane & 1);
        const int worker = lane >> 1;

        float score_max = -INFINITY;
        float sum_v = 0.0f;
        float sum_w = 0.0f;
        for (int j = worker; i0 < args.n_embd && j < 128; j += 16) {
            const int idx = idxs[j];
            if (idx < 0 || idx >= args.n_rows) {
                continue;
            }

            const float score = *(device const float *) (score_state + i0*args.nb_s0 + idx*args.nb_s1);
            const float value = *(device const float *) (kv_state + i0*args.nb_k0 + idx*args.nb_k1);
            if (score > score_max) {
                const float scale = fast::exp(score_max - score);
                sum_v = sum_v*scale + value;
                sum_w = sum_w*scale + 1.0f;
                score_max = score;
            } else {
                const float weight = fast::exp(score - score_max);
                sum_v += value*weight;
                sum_w += weight;
            }
        }

        float group_max = max(score_max, simd_shuffle_xor(score_max, 2));
        group_max = max(group_max, simd_shuffle_xor(group_max, 4));
        group_max = max(group_max, simd_shuffle_xor(group_max, 8));
        group_max = max(group_max, simd_shuffle_xor(group_max, 16));

        float scale = 0.0f;
        if (sum_w > 0.0f) {
            scale = fast::exp(score_max - group_max);
        }
        sum_v *= scale;
        sum_w *= scale;
        sum_v += simd_shuffle_xor(sum_v, 2);
        sum_w += simd_shuffle_xor(sum_w, 2);
        sum_v += simd_shuffle_xor(sum_v, 4);
        sum_w += simd_shuffle_xor(sum_w, 4);
        sum_v += simd_shuffle_xor(sum_v, 8);
        sum_w += simd_shuffle_xor(sum_w, 8);
        sum_v += simd_shuffle_xor(sum_v, 16);
        sum_w += simd_shuffle_xor(sum_w, 16);

        if (worker == 0 && i0 < args.n_embd) {
            *(device float *) (dst + i0*args.nb_d0 + ib*args.nb_d1) = sum_w > 0.0f ? sum_v/sum_w : 0.0f;
        }
        return;
    }

    const int i0 = tgpig.x*ntg.x + tiitg;
    if (i0 >= args.n_embd) {
        return;
    }

    float score_max = -INFINITY;
    float sum_v = 0.0f;
    float sum_w = 0.0f;
    for (int j = 0; j < n_read; ++j) {
        const int idx = idxs[j];
        if (idx < 0 || idx >= args.n_rows) {
            continue;
        }

        const bool cur_half = args.overlap && j >= args.ratio;
        const int i_src = (cur_half ? args.n_embd : 0) + i0;
        const float score = *(device const float *) (score_state + i_src*args.nb_s0 + idx*args.nb_s1);
        const float value = *(device const float *) (kv_state + i_src*args.nb_k0 + idx*args.nb_k1);
        if (score > score_max) {
            const float scale = fast::exp(score_max - score);
            sum_v = sum_v*scale + value;
            sum_w = sum_w*scale + 1.0f;
            score_max = score;
        } else {
            const float weight = fast::exp(score - score_max);
            sum_v += value*weight;
            sum_w += weight;
        }
    }

    *(device float *) (dst + i0*args.nb_d0 + ib*args.nb_d1) = sum_w > 0.0f ? sum_v/sum_w : 0.0f;
}

kernel void kernel_dsv4_top_k_mask(
        constant ggml_metal_kargs_dsv4_top_k_mask & args,
        device const char * raw_mask,
        device const char * comp_mask,
        device const char * comp_idx,
        device       char * dst,
        uint    tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort   ntg[[threads_per_threadgroup]]) {
    const int iq = tgpig % args.n_query;
    const int is = tgpig / args.n_query;

    device const half * rm = (device const half *) (raw_mask + (uint64_t) iq*args.nb_rm1 + (uint64_t) is*args.nb_rm3);
    device const half * cm = (device const half *) (comp_mask + (uint64_t) iq*args.nb_cm1 + (uint64_t) is*args.nb_cm3);
    device const int  * ci = (device const int  *) (comp_idx + (uint64_t) iq*args.nb_ci1 + (uint64_t) is*args.nb_ci3);
    device       half * out = (device half *) (dst + (uint64_t) iq*args.nb_d1 + (uint64_t) is*args.nb_d3);

    for (int i = tiitg; i < args.n_raw; i += ntg) {
        out[i] = rm[i];
    }
    for (int i = tiitg; i < args.n_comp; i += ntg) {
        out[args.n_raw + i] = -INFINITY;
    }
    threadgroup_barrier(mem_flags::mem_device);

    for (int i = tiitg; i < args.n_select; i += ntg) {
        const int idx = ci[i];
        if (idx >= 0 && idx < args.n_comp) {
            out[args.n_raw + idx] = cm[idx];
        }
    }
}

template <typename k4_t, short nl, void (*dequantize_func)(device const k4_t *, short, thread half4 &)>
kernel void kernel_dsv4_sparse_pack(
        constant ggml_metal_kargs_dsv4_sparse_pack & args,
        device const char * raw_k,
        device const char * comp_k,
        device const char * raw_mask,
        device const char * comp_mask,
        device const char * comp_idx,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort3  ntg[[threads_per_threadgroup]]) {
    constexpr int max_selected = 128 + 512;
    threadgroup int  selected_idx[max_selected];
    threadgroup half selected_mask[max_selected];

    if (args.n_raw == 0) {
        const int i  = tgpig.x;
        const int it = tgpig.y;
        const int iq = it % args.n_batch;
        const int is = it / args.n_batch;
        const int idx = *(device const int *) (comp_idx + (uint64_t) i*args.nb_ci0 + (uint64_t) iq*args.nb_ci1 + (uint64_t) is*args.nb_ci3);

        device const k4_t * src = (device const k4_t *) (comp_k + (uint64_t) idx*args.nb_ck2 + (uint64_t) is*args.nb_ck3);
        device half * out = (device half *) (dst + (uint64_t) it*args.nb_d1);
        device half4 * out_k = (device half4 *) out;
        for (int e4 = tiitg; e4 < args.n_embd/4; e4 += ntg.x) {
            half4 values;
            dequantize_func(src + e4/nl, e4%nl, values);
            out_k[i*args.n_embd/4 + e4] = values;
        }
        if (tiitg == 0) {
            out[args.n_embd*args.n_comp + i] = *(device const half *) (comp_mask + (uint64_t) idx*args.nb_cm0 + (uint64_t) iq*args.nb_cm1 + (uint64_t) is*args.nb_cm3);
        }
        return;
    }

    const int it = tgpig.x;
    const int iq = it % args.n_batch;
    const int is = it / args.n_batch;
    const int nk = args.n_raw + args.n_comp;

    device half * out = (device half *) (dst + (uint64_t) it*args.nb_d1);
    device half * out_k = out;
    device half * out_m = out + args.n_embd*nk;

    if (tiitg == 0) {
        int n = 0;
        for (int idx = 0; idx < args.n_raw_k && n < args.n_raw; ++idx) {
            const half m = *(device const half *) (raw_mask + (uint64_t) idx*args.nb_rm0 + (uint64_t) iq*args.nb_rm1 + (uint64_t) is*args.nb_rm3);
            if (isfinite(m)) {
                selected_idx[n] = idx;
                selected_mask[n] = m;
                ++n;
            }
        }
        for (; n < args.n_raw; ++n) {
            selected_idx[n] = -1;
            selected_mask[n] = -INFINITY;
        }
    }
    for (int i = tiitg; i < args.n_comp; i += ntg.x) {
        const int oi = args.n_raw + i;
        const int idx = *(device const int *) (comp_idx + (uint64_t) i*args.nb_ci0 + (uint64_t) iq*args.nb_ci1 + (uint64_t) is*args.nb_ci3);
        selected_idx[oi] = idx;
        selected_mask[oi] = *(device const half *) (comp_mask + (uint64_t) idx*args.nb_cm0 + (uint64_t) iq*args.nb_cm1 + (uint64_t) is*args.nb_cm3);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int i = 0; i < args.n_raw; ++i) {
        const int idx = selected_idx[i];
        if (idx >= 0) {
            device const k4_t * src = (device const k4_t *) (raw_k + (uint64_t) idx*args.nb_rk2 + (uint64_t) is*args.nb_rk3);
            for (int e4 = tiitg; e4 < args.n_embd/4; e4 += ntg.x) {
                half4 values;
                dequantize_func(src + e4/nl, e4%nl, values);
                ((device half4 *) out_k)[i*args.n_embd/4 + e4] = values;
            }
        } else {
            for (int e4 = tiitg; e4 < args.n_embd/4; e4 += ntg.x) {
                ((device half4 *) out_k)[i*args.n_embd/4 + e4] = 0.0h;
            }
        }
        if (tiitg == 0) {
            out_m[i] = selected_mask[i];
        }
    }

    for (int i = 0; i < args.n_comp; ++i) {
        const int oi = args.n_raw + i;
        const int idx = selected_idx[oi];
        device const k4_t * src = (device const k4_t *) (comp_k + (uint64_t) idx*args.nb_ck2 + (uint64_t) is*args.nb_ck3);
        for (int e4 = tiitg; e4 < args.n_embd/4; e4 += ntg.x) {
            half4 values;
            dequantize_func(src + e4/nl, e4%nl, values);
            ((device half4 *) out_k)[oi*args.n_embd/4 + e4] = values;
        }
        if (tiitg == 0) {
            out_m[oi] = selected_mask[oi];
        }
    }
}

typedef decltype(kernel_dsv4_sparse_pack<half4, 1, dequantize_f16_t4>) dsv4_sparse_pack_t;

template [[host_name("kernel_dsv4_sparse_pack_f16")]]
kernel dsv4_sparse_pack_t kernel_dsv4_sparse_pack<half4, 1, dequantize_f16_t4>;

template [[host_name("kernel_dsv4_sparse_pack_q8_0")]]
kernel dsv4_sparse_pack_t kernel_dsv4_sparse_pack<block_q8_0, 8, dequantize_q8_0_t4>;
