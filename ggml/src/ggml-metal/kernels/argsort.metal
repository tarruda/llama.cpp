#include "common.h"

// bitonic sort implementation following the CUDA kernels as reference
typedef void (argsort_t)(
        constant   ggml_metal_kargs_argsort & args,
        device   const char * src0,
        device      int32_t * dst,
        threadgroup int32_t * shmem_i32 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]);

template<ggml_sort_order order>
kernel void kernel_argsort_f32_i32(
        constant   ggml_metal_kargs_argsort & args,
        device   const char * src0,
        device      int32_t * dst,
        threadgroup int32_t * shmem_i32 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    // bitonic sort
    const int col = tpitg[0];
    const int ib  = tgpig[0] / args.ne01;

    const int i00 = ib*ntg.x;
    const int i01 = tgpig[0] % args.ne01;
    const int i02 = tgpig[1];
    const int i03 = tgpig[2];

    device const float * src0_row = (device const float *) (src0 + args.nb01*i01 + args.nb02*i02 + args.nb03*i03);

    // initialize indices
    shmem_i32[col] = i00 + col;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int k = 2; k <= ntg.x; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            int ixj = col ^ j;
            if (ixj > col) {
                if ((col & k) == 0) {
                    if (shmem_i32[col] >= args.ne00 ||
                       (shmem_i32[ixj] <  args.ne00 && (order == GGML_SORT_ORDER_ASC ?
                            src0_row[shmem_i32[col]] > src0_row[shmem_i32[ixj]] :
                            src0_row[shmem_i32[col]] < src0_row[shmem_i32[ixj]]))
                    ) {
                        SWAP(shmem_i32[col], shmem_i32[ixj]);
                    }
                } else {
                    if (shmem_i32[ixj] >= args.ne00 ||
                       (shmem_i32[col] <  args.ne00 && (order == GGML_SORT_ORDER_ASC ?
                            src0_row[shmem_i32[col]] < src0_row[shmem_i32[ixj]] :
                            src0_row[shmem_i32[col]] > src0_row[shmem_i32[ixj]]))
                    ) {
                        SWAP(shmem_i32[col], shmem_i32[ixj]);
                    }
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    const int64_t i0 = ib*args.top_k;

    // copy the result to dst without the padding
    if (i0 + col < args.ne0 && col < args.top_k) {
        dst += i0 + args.ne0*i01 + args.ne0*args.ne1*i02 + args.ne0*args.ne1*args.ne2*i03;

        dst[col] = shmem_i32[col];
    }
}

template [[host_name("kernel_argsort_f32_i32_asc")]]  kernel argsort_t kernel_argsort_f32_i32<GGML_SORT_ORDER_ASC>;
template [[host_name("kernel_argsort_f32_i32_desc")]] kernel argsort_t kernel_argsort_f32_i32<GGML_SORT_ORDER_DESC>;

kernel void kernel_top_k_512_10_f32_i32(
        constant ggml_metal_kargs_argsort & args,
        device const char    * src0,
        device       int32_t * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]]) {
    constexpr ushort n_per_lane = 16;
    constexpr ushort n_top = 10;

    const int i01 = tgpig.x;
    const int i02 = tgpig.y;
    const int i03 = tgpig.z;
    device const float * row = (device const float *) (src0 + args.nb01*i01 + args.nb02*i02 + args.nb03*i03);

    float values[n_per_lane];
    bool active[n_per_lane];
    for (ushort i = 0; i < n_per_lane; ++i) {
        values[i] = row[tiisg + i*N_SIMDWIDTH];
        active[i] = true;
    }

    dst += args.ne0*i01 + args.ne0*args.ne1*i02 + args.ne0*args.ne1*args.ne2*i03;

    for (ushort k = 0; k < n_top; ++k) {
        float lmax = -INFINITY;
        int32_t larg = -1;
        for (ushort i = 0; i < n_per_lane; ++i) {
            const int32_t index = tiisg + i*N_SIMDWIDTH;
            if (active[i] && (larg < 0 || values[i] > lmax || (values[i] == lmax && index > larg))) {
                lmax = values[i];
                larg = index;
            }
        }

        const float max_val = simd_max(lmax);
        const int32_t arg_val = simd_max(select(-1, larg, lmax == max_val));

        if (tiisg == 0) {
            dst[k] = arg_val;
        }
        if (tiisg == arg_val % N_SIMDWIDTH) {
            active[arg_val / N_SIMDWIDTH] = false;
        }
    }
}

kernel void kernel_top_k_radix_f32_i32(
        constant ggml_metal_kargs_argsort & args,
        device const char    * src0,
        device       int32_t * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiitg[[thread_index_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    constexpr ushort radix = 16;
    constexpr ushort n_top = 512;

    threadgroup atomic_uint histogram[radix];
    threadgroup atomic_uint n_greater;
    threadgroup atomic_uint n_equal;
    threadgroup uint prefix;
    threadgroup uint rank;
    threadgroup int32_t selected[n_top];

    const int i01 = tgpig.x;
    const int i02 = tgpig.y;
    const int i03 = tgpig.z;
    device const float * row = (device const float *) (src0 + args.nb01*i01 + args.nb02*i02 + args.nb03*i03);

    if (tiitg == 0) {
        prefix = 0;
        rank = n_top - 1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Select the exact top-k partition with an MSD radix pass.
    for (int shift = 28; shift >= 0; shift -= 4) {
        if (tiitg < radix) {
            atomic_store_explicit(&histogram[tiitg], 0u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint mask = shift == 28 ? 0u : 0xffffffffu << (shift + 4);
        for (int i = tiitg; i < args.ne00; i += ntg.x) {
            const uint bits = as_type<uint>(row[i]);
            const uint key = bits ^ (uint(int(bits) >> 31) | 0x80000000u);
            if ((key & mask) == prefix) {
                atomic_fetch_add_explicit(&histogram[(key >> shift) & 0xfu], 1u, memory_order_relaxed);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiitg == 0) {
            uint r = rank;
            for (int digit = radix - 1; digit >= 0; --digit) {
                const uint count = atomic_load_explicit(&histogram[digit], memory_order_relaxed);
                if (r < count) {
                    prefix |= uint(digit) << shift;
                    rank = r;
                    break;
                }
                r -= count;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tiitg == 0) {
        atomic_store_explicit(&n_greater, 0u, memory_order_relaxed);
        atomic_store_explicit(&n_equal, 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int i = tiitg; i < args.ne00; i += ntg.x) {
        const uint bits = as_type<uint>(row[i]);
        const uint key = bits ^ (uint(int(bits) >> 31) | 0x80000000u);
        if (key > prefix) {
            const uint pos = atomic_fetch_add_explicit(&n_greater, 1u, memory_order_relaxed);
            selected[pos] = i;
        } else if (key == prefix) {
            const uint pos = n_top - rank - 1 + atomic_fetch_add_explicit(&n_equal, 1u, memory_order_relaxed);
            if (pos < n_top) {
                selected[pos] = i;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    dst += args.ne0*i01 + args.ne0*args.ne1*i02 + args.ne0*args.ne1*args.ne2*i03;
    for (int i = tiitg; i < n_top; i += ntg.x) {
        dst[i] = selected[i];
    }
}

typedef void (argsort_merge_t)(
        constant   ggml_metal_kargs_argsort_merge & args,
        device const char    * src0,
        device const int32_t * tmp,
        device       int32_t * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]);

template<ggml_sort_order order>
kernel void kernel_argsort_merge_f32_i32(
        constant   ggml_metal_kargs_argsort_merge & args,
        device const char    * src0,
        device const int32_t * tmp,
        device       int32_t * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {

    const int im  = tgpig[0] / args.ne01;
    const int i01 = tgpig[0] % args.ne01;
    const int i02 = tgpig[1];
    const int i03 = tgpig[2];

    const int start = im * (2 * args.len);

    const int len0 = MIN(args.len, MAX(0, args.ne0 - (int)(start)));
    const int len1 = MIN(args.len, MAX(0, args.ne0 - (int)(start + args.len)));

    const int total = len0 + len1;

    device const int32_t * tmp0 = tmp + start
        + i01*args.ne0
        + i02*args.ne0*args.ne01
        + i03*args.ne0*args.ne01*args.ne02;

    device const int32_t * tmp1 = tmp0 + args.len;

    dst += start
        + i01*args.top_k
        + i02*args.top_k*args.ne01
        + i03*args.top_k*args.ne01*args.ne02;

    device const float * src0_row = (device const float *)(src0
        + args.nb01*i01
        + args.nb02*i02
        + args.nb03*i03);

    if (total == 0) {
        return;
    }

    const int chunk = (total + ntg.x - 1) / ntg.x;

    const int k0 = tpitg.x * chunk;
    const int k1 = MIN(MIN(k0 + chunk, total), args.top_k);

    if (k0 >= args.top_k) {
        return;
    }

    if (k0 >= total) {
        return;
    }

    int low  = k0 > len1 ? k0 - len1 : 0;
    int high = MIN(k0, len0);

    // binary-search partition (i, j) such that i + j = k
    while (low < high) {
        const int mid = (low + high) >> 1;

        const int32_t idx0 = tmp0[mid];
        const int32_t idx1 = tmp1[k0 - mid - 1];

        const float val0 = src0_row[idx0];
        const float val1 = src0_row[idx1];

        bool take_left;
        if (order == GGML_SORT_ORDER_ASC) {
            take_left = (val0 <= val1);
        } else {
            take_left = (val0 >= val1);
        }

        if (take_left) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }

    int i = low;
    int j = k0 - i;

    // keep the merge fronts into registers
    int32_t idx0 = 0;
    float   val0 = 0.0f;
    if (i < len0) {
        idx0 = tmp0[i];
        val0 = src0_row[idx0];
    }

    int32_t idx1 = 0;
    float   val1 = 0.0f;
    if (j < len1) {
        idx1 = tmp1[j];
        val1 = src0_row[idx1];
    }

    for (int k = k0; k < k1; ++k) {
        int32_t out_idx;

        if (i >= len0) {
            while (k < k1) {
                dst[k++] = tmp1[j++];
            }
            break;
        } else if (j >= len1) {
            while (k < k1) {
                dst[k++] = tmp0[i++];
            }
            break;
        } else {
            bool take_left;

            if (order == GGML_SORT_ORDER_ASC) {
                take_left = (val0 <= val1);
            } else {
                take_left = (val0 >= val1);
            }

            if (take_left) {
                out_idx = idx0;
                ++i;
                if (i < len0) {
                    idx0 = tmp0[i];
                    val0 = src0_row[idx0];
                }
            } else {
                out_idx = idx1;
                ++j;
                if (j < len1) {
                    idx1 = tmp1[j];
                    val1 = src0_row[idx1];
                }
            }
        }

        dst[k] = out_idx;
    }
}

template [[host_name("kernel_argsort_merge_f32_i32_asc")]]  kernel argsort_merge_t kernel_argsort_merge_f32_i32<GGML_SORT_ORDER_ASC>;
template [[host_name("kernel_argsort_merge_f32_i32_desc")]] kernel argsort_merge_t kernel_argsort_merge_f32_i32<GGML_SORT_ORDER_DESC>;
