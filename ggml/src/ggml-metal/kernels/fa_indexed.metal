#include "common.h"
#include "dequantize.h"

// dequantize a quantized KV cache tensor to contiguous F16 before running the F16 flash attention kernels
// - one thread per block; dispatched separately for K and V
// - ref: https://github.com/ggml-org/llama.cpp/pull/27390
template <
    typename block_t,
    short QK,
    void (*deq_t4x4)(device const block_t *, short, thread float4x4 &)>
kernel void kernel_flash_attn_ext_kv_f16(
        constant ggml_metal_kargs_flash_attn_ext_kv_f16 & args,
        device const char * x,
        device       half * x_dst,
        uint gid [[thread_position_in_grid]]) {
    if (gid >= (uint) args.nblocks) {
        return;
    }

    const uint nb = args.ne0/QK;
    const uint i0 = gid%nb;
    uint ib       = gid/nb;
    const uint i1 = ib%args.ne1;
    ib /= args.ne1;
    const uint i2 = ib%args.ne2;
    const uint i3 = ib/args.ne2;

    const uint64_t offs = i0*args.nb0 + i1*args.nb1 + i2*args.nb2 + i3*args.nb3;

    device const block_t * src = (device const block_t *) (x + offs);
    device half4 * dst = (device half4 *) x_dst + (QK/4)*gid;

    for (short i = 0; i < QK/16; ++i) {
        float4x4 reg;
        deq_t4x4(src, i, reg);
        dst[4*i + 0] = (half4) reg[0];
        dst[4*i + 1] = (half4) reg[1];
        dst[4*i + 2] = (half4) reg[2];
        dst[4*i + 3] = (half4) reg[3];
    }
}

typedef decltype(kernel_flash_attn_ext_kv_f16<block_q8_0, 32, dequantize_q8_0>) kernel_flash_attn_ext_kv_f16_t;

template [[host_name("kernel_flash_attn_ext_kv_q4_0_f16")]] kernel kernel_flash_attn_ext_kv_f16_t kernel_flash_attn_ext_kv_f16<block_q4_0, 32, dequantize_q4_0>;
template [[host_name("kernel_flash_attn_ext_kv_q4_1_f16")]] kernel kernel_flash_attn_ext_kv_f16_t kernel_flash_attn_ext_kv_f16<block_q4_1, 32, dequantize_q4_1>;
template [[host_name("kernel_flash_attn_ext_kv_q5_0_f16")]] kernel kernel_flash_attn_ext_kv_f16_t kernel_flash_attn_ext_kv_f16<block_q5_0, 32, dequantize_q5_0>;
template [[host_name("kernel_flash_attn_ext_kv_q5_1_f16")]] kernel kernel_flash_attn_ext_kv_f16_t kernel_flash_attn_ext_kv_f16<block_q5_1, 32, dequantize_q5_1>;
template [[host_name("kernel_flash_attn_ext_kv_q8_0_f16")]] kernel kernel_flash_attn_ext_kv_f16_t kernel_flash_attn_ext_kv_f16<block_q8_0, 32, dequantize_q8_0>;

kernel void kernel_flash_attn_ext_indexed_pack_f16_d256(
        constant ggml_metal_kargs_flash_attn_ext_indexed & args,
        device const char * k,
        device const char * v,
        device const char * indices,
        device const char * mask,
        device       char * dst,
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]]) {
    const int32_t ir = tgpig.x;
    const int32_t is = tgpig.y;
    const int64_t kv_stride = sizeof(half)*OP_FLASH_ATTN_EXT_INDEXED_D*args.n_padded*OP_FLASH_ATTN_EXT_INDEXED_N_KV;
    const int64_t stream_stride = 2*kv_stride + sizeof(half)*args.n_padded;

    device const int32_t * pi = (device const int32_t *) (indices + is*args.nb_i3);
    device const half * pm = (device const half *) (mask + is*args.nb_m3);
    const int32_t idx = ir < args.n_select ? pi[ir] : -1;
    const half mask_value = ir < args.n_select ? pm[ir] : half(-INFINITY);
    const bool valid = idx >= 0 && idx < args.n_kv && float(mask_value) != -INFINITY;
    const int32_t safe_idx = valid ? idx : 0;

    device half * pk_dst = (device half *) (dst + is*stream_stride);
    device half * pv_dst = (device half *) (dst + is*stream_stride + kv_stride);
    FOR_UNROLL (short ih = 0; ih < OP_FLASH_ATTN_EXT_INDEXED_N_KV; ++ih) {
        device const half * pk_src = (device const half *) (k + safe_idx*args.nb_k1 + ih*args.nb_k2 + is*args.nb_k3);
        device const half * pv_src = (device const half *) (v + safe_idx*args.nb_v1 + ih*args.nb_v2 + is*args.nb_v3);
        const int64_t offs = ((int64_t) ih*args.n_padded + ir)*OP_FLASH_ATTN_EXT_INDEXED_D + tiitg;
        pk_dst[offs] = valid ? pk_src[tiitg] : half(0.0f);
        pv_dst[offs] = valid ? pv_src[tiitg] : half(0.0f);
    }

    if (tiitg == 0) {
        device half * pm_dst = (device half *) (dst + is*stream_stride + 2*kv_stride);
        pm_dst[ir] = valid ? mask_value : half(-INFINITY);
    }
}
