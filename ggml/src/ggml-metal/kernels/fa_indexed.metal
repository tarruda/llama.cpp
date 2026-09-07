#include "common.h"
#include "dequantize.h"

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
