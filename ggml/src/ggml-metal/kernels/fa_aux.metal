#include "common.h"
#include "dequantize.h"

constant bool FC_flash_attn_ext_pad_has_mask [[function_constant(FC_FLASH_ATTN_EXT_PAD + 0)]];

constant int32_t FC_flash_attn_ext_pad_ncpsg [[function_constant(FC_FLASH_ATTN_EXT_PAD + 25)]];

// pad the last chunk of C elements of k and v into a an extra pad buffer
kernel void kernel_flash_attn_ext_pad(
        constant ggml_metal_kargs_flash_attn_ext_pad & args,
        device const char * k,
        device const char * v,
        device const char * mask,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiitg[[thread_index_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int32_t C = FC_flash_attn_ext_pad_ncpsg;

    device char * k_pad    = dst;
    device char * v_pad    = k_pad + args.nb11*C*args.ne_12_2*args.ne_12_3;
    device char * mask_pad = v_pad + args.nb21*C*args.ne_12_2*args.ne_12_3;

    const int32_t icp = args.ne11 % C;
    const int32_t ic0 = args.ne11 - icp;

    const int32_t i1 = tgpig[0];
    const int32_t i2 = tgpig[1];
    const int32_t i3 = tgpig[2];

    if (i2 < args.ne_12_2 && i3 < args.ne_12_3) {
        device const char * k_src = k + args.nb11*(ic0 + i1) + args.nb12*i2 + args.nb13*i3;
        device const char * v_src = v + args.nb21*(ic0 + i1) + args.nb22*i2 + args.nb23*i3;

        device char * k_dst = k_pad + args.nb11*i1 + args.nb11*C*i2 + args.nb11*C*args.ne_12_2*i3;
        device char * v_dst = v_pad + args.nb21*i1 + args.nb21*C*i2 + args.nb21*C*args.ne_12_2*i3;

        if (i1 >= icp) {
            // here it is not important the exact value that will be used as we rely on masking out the scores in the attention
            for (uint64_t i = tiitg; i < args.nb11; i += ntg.x) {
                k_dst[i] = 0;
            }
            for (uint64_t i = tiitg; i < args.nb21; i += ntg.x) {
                v_dst[i] = 0;
            }
        } else {
            for (uint64_t i = tiitg; i < args.nb11; i += ntg.x) {
                k_dst[i] = k_src[i];
            }
            for (uint64_t i = tiitg; i < args.nb21; i += ntg.x) {
                v_dst[i] = v_src[i];
            }
        }
    }

    if (FC_flash_attn_ext_pad_has_mask) {
        if (i2 < args.ne32 && i3 < args.ne33) {
            for (int ib = i1; ib < args.ne31; ib += C) {
                device const half * mask_src = (device const half *)(mask      + args.nb31*ib + args.nb32*i2 + args.nb33*i3) + ic0;
                device       half * mask_dst = (device       half *)(mask_pad) + C*ib + C*args.ne31*i2 + C*args.ne31*args.ne32*i3;

                for (int i = tiitg; i < C; i += ntg.x) {
                    if (i >= icp) {
                        mask_dst[i] = -MAXHALF;
                    } else {
                        mask_dst[i] = mask_src[i];
                    }
                }
            }
        }
    }
}

constant int32_t FC_flash_attn_ext_blk_nqptg [[function_constant(FC_FLASH_ATTN_EXT_BLK + 24)]];
constant int32_t FC_flash_attn_ext_blk_ncpsg [[function_constant(FC_FLASH_ATTN_EXT_BLK + 25)]];

// scan the blocks of the mask that are not masked
// 0 -     masked (i.e. full of -INF, skip)
// 1 - not masked (i.e. at least one element of the mask is not -INF)
// 2 - all zero
kernel void kernel_flash_attn_ext_blk(
        constant ggml_metal_kargs_flash_attn_ext_blk & args,
        device const char * mask,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]]) {
    // block size C x Q
    const int32_t Q = FC_flash_attn_ext_blk_nqptg;
    const int32_t C = FC_flash_attn_ext_blk_ncpsg;

    constexpr short NW  = N_SIMDWIDTH;

    const int32_t i3 = tgpig[2]/args.ne32;
    const int32_t i2 = tgpig[2]%args.ne32;
    const int32_t i1 = tgpig[1];
    const int32_t i0 = tgpig[0];

    char res = i0*C + C > args.ne30 || i1*Q + Q > args.ne31 ? 1 : 0;

    // detailed check of the elements of the block
    if ((C > NW || Q > 1) && res == 0) {
        half mmin =  MAXHALF;
        half mmax = -MAXHALF;

        const short nq_mask = args.ne31 == 1 ? 1 : Q;
        FOR_UNROLL (short j = 0; j < nq_mask; ++j) {
            device const half * mask_src = (device const half *) (mask + ((i1*Q + j)%args.ne31)*args.nb31 + i2*args.nb32 + i3*args.nb33) + i0*C + tiisg;
            FOR_UNROLL (short ii = 0; ii < C/NW; ++ii) {
                mmin = min(mmin, mask_src[ii*NW]);
                mmax = max(mmax, mask_src[ii*NW]);
            }
        }

        mmin = simd_min(mmin);
        mmax = simd_max(mmax);

        if (mmax > -MAXHALF) {
            if (mmin == 0.0 && mmax == 0.0) {
                res = 2;
            } else {
                res = 1;
            }
        }
    }

    const int32_t nblk1 = ((args.ne01 + Q - 1)/Q);
    const int32_t nblk0 = ((args.ne30 + C - 1)/C);

    if (tiisg == 0) {
        dst[((i3*args.ne32 + i2)*nblk1 + i1)*nblk0 + i0] = res;
    }
}
// compress the finite entries of each KQ mask row into a list of KV indices (ascending order),
// padded with -1 up to n_kv_max_padded (a multiple of OP_FLASH_ATTN_EXT_VEC_NCPSG)
// one threadgroup per mask row; the mask remains the single source of truth for the values
kernel void kernel_flash_attn_ext_vec_idx(
        constant ggml_metal_kargs_flash_attn_ext_vec_idx & args,
        device const half * mask,
        device       int  * idx,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiitg[[thread_index_in_threadgroup]],
        ushort3 ntg[[threads_per_threadgroup]]) {
    constexpr short NW = N_SIMDWIDTH;
    constexpr short NLOCAL = 32; // max finite positions kept in registers per thread

    const int i1 = tgpig[0];
    const int i2 = tgpig[1];
    const int i3 = tgpig[2];

    device const half * pm  = (device const half *) ((device const char *) mask + i1*args.nb31 + i2*args.nb32 + i3*args.nb33);
    device int * pidx = idx + (((int64_t)i3*args.ne32 + i2)*args.ne31 + i1)*args.n_kv_max_padded;

    const int n  = args.ne30;
    const int q  = n/ntg.x;
    const int r  = n%ntg.x;

    // each thread handles a contiguous slice of the mask row
    const int r0 = q*tiitg + min((int) tiitg, r);
    const int r1 = r0 + q + (tiitg < r ? 1 : 0);

    // count the finite entries in the slice and keep their positions in registers (single mask read)
    int cnt = 0;  // total finite entries in the slice
    int nloc = 0; // finite entries kept in registers
    int local[NLOCAL];
    for (int i = r0; i < r1; ++i) {
        if (isfinite((float) pm[i])) {
            if (nloc < NLOCAL) {
                local[nloc] = i;
                nloc++;
            }
            cnt++;
        }
    }

    const short sgitg = tiitg/NW;
    const short tiisg = tiitg%NW;

    threadgroup int tcount[8];

    // simd_sum is a collective: all lanes must evaluate it
    const int sg_sum = simd_sum(cnt);
    if (tiisg == 0) {
        tcount[sgitg] = sg_sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    int total = 0;
    for (short s = 0; s < ntg.x/NW; ++s) {
        total += tcount[s];
    }

    // base offset of this thread's slice in the output list (exclusive scan within the simdgroup)
    int sg_base = 0;
    for (short s = 0; s < sgitg; ++s) {
        sg_base += tcount[s];
    }

    // exclusive prefix scan of the per-thread counts within the simdgroup
    int incl = cnt;
    for (int d = 1; d < NW; d <<= 1) {
        const int v = simd_shuffle_up(incl, d);
        if (tiisg >= d) {
            incl += v;
        }
    }
    const int base = sg_base + (incl - cnt);

    // write the finite positions in order; if the hint is violated, keep only the first n_kv_max entries
    int j = 0;
    for (; j < nloc && base + j < args.n_kv_max; ++j) {
        pidx[base + j] = local[j];
    }

    // a dense mask may have more than NLOCAL finite entries in a slice; re-read the mask to write the rest
    if (cnt > nloc && base + nloc < args.n_kv_max) {
        int j2 = 0;
        for (int i = r0; i < r1; ++i) {
            if (isfinite((float) pm[i])) {
                if (j2 >= nloc) {
                    pidx[base + j2] = i;
                }
                j2++;
                if (base + j2 >= args.n_kv_max) {
                    break;
                }
            }
        }
    }

    // pad the tail of the list with -1
    const int count = min(total, args.n_kv_max);
    for (int i = count + tiitg; i < args.n_kv_max_padded; i += ntg.x) {
        pidx[i] = -1;
    }
}

constant int32_t FC_flash_attn_ext_vec_reduce_DV  [[function_constant(FC_FLASH_ATTN_EXT_VEC_REDUCE + 0)]];
constant int32_t FC_flash_attn_ext_vec_reduce_NWG [[function_constant(FC_FLASH_ATTN_EXT_VEC_REDUCE + 1)]];

kernel void kernel_flash_attn_ext_vec_reduce(
        constant ggml_metal_kargs_flash_attn_ext_vec_reduce & args,
        device  const char * htmp,
        device        char * dst,
        uint   tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
#define NWG (FC_flash_attn_ext_vec_reduce_NWG)
#define DV  (FC_flash_attn_ext_vec_reduce_DV)

    const uint64_t rid = tgpig;

    const short iwg = tiisg;

    device const float  * ss    = (device const float  *) htmp + (uint64_t)args.nrows*DV*NWG;

    float S = ss[rid*(2*NWG) + 2*iwg + 0];
    float M = ss[rid*(2*NWG) + 2*iwg + 1];

    const float m  = simd_max(M);
    const float ms = exp(M - m);

    S = simd_sum(S*ms);
    S = S == 0.0f ? 0.0f : 1.0f/S;

    const short DV4 = DV/4;

    device const float4 * htmp4 = (device const float4 *) htmp + rid*DV4*NWG;
    device       float4 * dst4  = (device       float4 *) dst  + rid*DV4;

    for (short i = sgitg; i < DV4; i += NWG) {
        const float4 v = simd_sum(htmp4[i*NWG + iwg]*ms);

        if (iwg == 0) {
            dst4[i] = v*S;
        }
    }

#undef NWG
#undef DV
}

constant short FC_lightning_indexer_nh [[function_constant(FC_LIGHTNING_INDEXER + 0)]];

template<typename k4_t>
kernel void kernel_lightning_indexer_tiled(
        constant ggml_metal_kargs_lightning_indexer & args,
        device const char * q,
        device const char * k,
        device const char * w,
        device const char * m,
        device char * dst,
        threadgroup float * shared [[threadgroup(0)]],
        uint3 group [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup float * qs = shared;
    threadgroup float * ks = qs + 32*64;
    const uint tx = tid%8;
    const uint ty = tid/8;
    const uint iq = group.y*16 + ty;
    const uint ik = group.x*32 + tx*4;
    float4 acc[4] = {};
    for (uint kc = 0; kc < 128; kc += 32) {
        FOR_UNROLL (uint j = 0; j < 4; ++j) {
            const uint i = tid + j*128;
            const uint row = i/8;
            const uint col = 4*(i%8);
            const uint query = group.y*16 + row/4;
            float4 v = 0.0f;
            if (query < args.n_batch) {
                v = *(device const float4 *) (q + group.z*args.nbq3 + query*args.nbq2 + (row%4)*args.nbq1 + (kc + col)*sizeof(float));
            }
            FOR_UNROLL (uint x = 0; x < 4; ++x) {
                qs[(col + x)*64 + row] = v[x];
            }
        }
        FOR_UNROLL (uint j = 0; j < 2; ++j) {
            const uint i = tid + j*128;
            const uint row = i/8;
            const uint col = 4*(i%8);
            const uint key = group.x*32 + row;
            float4 v = 0.0f;
            if (key < args.n_kv) {
                v = float4(*(device const k4_t *) (k + group.z*args.nbk3 + key*args.nbk2 + (kc + col)*sizeof(k4_t)/4));
            }
            FOR_UNROLL (uint x = 0; x < 4; ++x) {
                ks[(col + x)*32 + row] = v[x];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint x = 0; x < 32; ++x) {
            const float4 qv = *(threadgroup float4 *) (qs + x*64 + ty*4);
            const float4 kv = *(threadgroup float4 *) (ks + x*32 + tx*4);
            FOR_UNROLL (uint h = 0; h < 4; ++h) {
                acc[h] = fma(float4(qv[h]), kv, acc[h]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (iq < args.n_batch) {
        device const float * weights = (device const float *) (w + iq*args.nbw1 + group.z*args.nbw3);
        device const half * mask = (device const half *) (m + iq*args.nbm1 + (group.z%args.mask_ne3)*args.nbm3);
        float4 score = 0.0f;
        FOR_UNROLL (uint h = 0; h < 4; ++h) {
            score += max(acc[h], 0.0f)*weights[h];
        }
        FOR_UNROLL (uint j = 0; j < 4; ++j) {
            if (ik + j < args.n_kv) {
                *(device float *) (dst + (ik + j)*sizeof(float) + iq*args.nb1 + group.z*args.nb3) = score[j] + float(mask[ik + j]);
            }
        }
    }
}

typedef decltype(kernel_lightning_indexer_tiled<float4>) kernel_lightning_indexer_tiled_t;
template [[host_name("kernel_lightning_indexer_tiled_f32")]] kernel kernel_lightning_indexer_tiled_t kernel_lightning_indexer_tiled<float4>;
template [[host_name("kernel_lightning_indexer_tiled_f16")]] kernel kernel_lightning_indexer_tiled_t kernel_lightning_indexer_tiled<half4>;

template<typename k4_t>
kernel void kernel_lightning_indexer_vec(
        constant ggml_metal_kargs_lightning_indexer & args,
        device const char * q,
        device const char * k,
        device const char * w,
        device const char * m,
        device       char * dst,
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const int iq = tgpig.y;
    const int is = tgpig.z;
    device const char * q_row = q + iq*args.nbq2 + is*args.nbq3 + tiisg*sizeof(float4);
    device const float * weights = (device const float *) (w + iq*args.nbw1 + is*args.nbw3);
    float4 q_reg[OP_LIGHTNING_INDEXER_VEC_NH];
    FOR_UNROLL (ushort ih = 0; ih < OP_LIGHTNING_INDEXER_VEC_NH; ++ih) {
        q_reg[ih] = *(device const float4 *) (q_row + ih*args.nbq1);
    }

    const int ib0 = (tgpig.x*OP_LIGHTNING_INDEXER_VEC_NSG + sgitg)*OP_LIGHTNING_INDEXER_VEC_NKPSG;
    FOR_UNROLL (ushort ik = 0; ik < OP_LIGHTNING_INDEXER_VEC_NKPSG; ++ik) {
        const int ib = ib0 + ik;
        if (ib >= args.n_kv) {
            return;
        }
        device const k4_t * k_row = (device const k4_t *) (k + ib*args.nbk2 + is*args.nbk3);
        const float4 kv = float4(k_row[tiisg]);
        float score = 0.0f;
        FOR_UNROLL (ushort ih = 0; ih < OP_LIGHTNING_INDEXER_VEC_NH; ++ih) {
            score += max(simd_sum(dot(q_reg[ih], kv)), 0.0f)*weights[ih];
        }
        if (tiisg == 0) {
            device const half * mask = (device const half *) (m + iq*args.nbm1 + (is % args.mask_ne3)*args.nbm3);
            *(device float *) (dst + ib*sizeof(float) + iq*args.nb1 + is*args.nb3) = score + float(mask[ib]);
        }
    }
}

typedef decltype(kernel_lightning_indexer_vec<float4>) kernel_lightning_indexer_vec_t;
template [[host_name("kernel_lightning_indexer_vec_f32")]] kernel kernel_lightning_indexer_vec_t kernel_lightning_indexer_vec<float4>;
template [[host_name("kernel_lightning_indexer_vec_f16")]] kernel kernel_lightning_indexer_vec_t kernel_lightning_indexer_vec<half4>;

template<
    typename kd4x4_t,
    short nl_k,
    void (*deq_k)(device const kd4x4_t *, short, thread half4x4 &),
    bool direct_k>
kernel void kernel_lightning_indexer(
        constant ggml_metal_kargs_lightning_indexer & args,
        device const char * q,
        device const char * k,
        device const char * w,
        device const char * m,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    constexpr short DK    = OP_LIGHTNING_INDEXER_DK;
    const     short NH    = FC_lightning_indexer_nh;
    constexpr short NHPTG = OP_LIGHTNING_INDEXER_NHPTG;
    constexpr short NKPSG = OP_LIGHTNING_INDEXER_NKPSG;
    constexpr short NSG   = OP_LIGHTNING_INDEXER_NSG;
    constexpr short NBPTG = OP_LIGHTNING_INDEXER_NBPTG;

    constexpr short DK4  = DK/4;
    constexpr short DK8  = DK/8;
    constexpr short DK16 = DK/16;

    constexpr short NK  = NKPSG*NSG; // keys    per threadgroup
    constexpr short NTG = 32*NSG;    // threads per threadgroup

    const int i_stream = tgpig.z;
    const int i_kv_0   = args.kv_offset + tgpig.x*NK; // first key of this threadgroup
    const int i_kv     = i_kv_0 + sgitg*NKPSG;  // first key of this simdgroup

    simdgroup_half8x8 mk[DK8];

    threadgroup half sk[direct_k ? 1 : NK * DK16 * 16];
    threadgroup half4x4 * sk4x4 = (threadgroup half4x4 *) sk;

    if (direct_k) {
        device const half * pk = (device const half *) (k + i_kv*args.nbk2 + i_stream*args.nbk3);

        FOR_UNROLL (short i = 0; i < DK8; ++i) {
            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_load(mk[i], pk + 8*i, args.nbk2/sizeof(half), 0, true);
            simdgroup_barrier(mem_flags::mem_none);
        }
    } else {
        for (short i = tiitg; i < NK*DK16; i += NTG) {
            const short ik  = i/DK16;
            const short i16 = i%DK16;

            half4x4 tmp;

            if (i_kv_0 + ik < args.n_kv) {
                device const kd4x4_t * kr = (device const kd4x4_t *) (k + (i_kv_0 + ik)*args.nbk2 + i_stream*args.nbk3);

                deq_k(kr + i16/nl_k, i16%nl_k, tmp);
            } else {
                FOR_UNROLL (short j = 0; j < 4; ++j) {
                    tmp[j] = half4(0.0h);
                }
            }

            sk4x4[i] = tmp;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // K tile of this simdgroup, transposed to [DK, NKPSG]
        FOR_UNROLL (short i = 0; i < DK8; ++i) {
            simdgroup_load(mk[i], sk + sgitg*NKPSG*DK + 8*i, DK, 0, true);
        }
    }

    threadgroup half4   sq4[NHPTG*DK4];
    threadgroup half  * sq = (threadgroup half *) sq4;

    threadgroup float sw [NHPTG];
    threadgroup float sqk[NSG*NHPTG*NKPSG];

    const int i_batch_0 = tgpig.y*NBPTG;
    const int n_batch   = min((int) NBPTG, args.n_batch - i_batch_0);

    for (short ib = 0; ib < n_batch; ++ib) {
        const int i_batch = i_batch_0 + ib;

        device const char * pq = q + i_batch*args.nbq2 + i_stream*args.nbq3;
        device const char * pw = w + i_batch*args.nbw1 + i_stream*args.nbw3;

        float score = 0.0f;

        FOR_UNROLL (short i_head = 0; i_head < NH; i_head += NHPTG) {
            // stage the Q tile [DK, NHPTG] and the (prescaled) head weights, heads past NH are zero
            for (short i = tiitg; i < NHPTG*DK4; i += NTG) {
                const short ih = i/DK4;
                const short i4 = i%DK4;

                if (i_head + ih < NH) {
                    device const float4 * q4 = (device const float4 *) (pq + (i_head + ih)*args.nbq1);

                    sq4[ih*DK4 + i4] = half4(q4[i4]);
                } else {
                    sq4[ih*DK4 + i4] = half4(0.0h);
                }
            }

            if (tiitg < NHPTG) {
                sw[tiitg] = i_head + tiitg < NH ? ((device const float *) pw)[i_head + tiitg] : 0.0f;
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            simdgroup_float8x8 mqk = make_filled_simdgroup_matrix<float, 8>(0.0f);

            FOR_UNROLL (short i = 0; i < DK8; ++i) {
                simdgroup_half8x8 mq;

                simdgroup_load(mq, sq + 8*i, DK, 0, false);
                simdgroup_multiply_accumulate(mqk, mq, mk[i], mqk);
            }

            threadgroup float * pqk = sqk + sgitg*NHPTG*NKPSG;

            simdgroup_store(mqk, pqk, NKPSG, 0, false);
            simdgroup_barrier(mem_flags::mem_threadgroup);

            // one lane per key: ReLU, apply the head weight and accumulate over the head tile
            if (tiisg < NKPSG) {
                FOR_UNROLL (short ih = 0; ih < NHPTG; ++ih) {
                    score += max(pqk[ih*NKPSG + tiisg], 0.0f)*sw[ih];
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (tiisg < NKPSG) {
            const int ik = i_kv + tiisg;
            if (ik < args.n_kv) {
                device const half  * pm = (device const half  *) (m   + i_batch*args.nbm1 + (i_stream % args.mask_ne3)*args.nbm3);
                device       float * pd = (device       float *) (dst + i_batch*args.nb1  + i_stream*args.nb3);

                pd[ik] = score + (float) pm[ik];
            }
        }
    }
}

typedef decltype(kernel_lightning_indexer<half4x4, 1, dequantize_f16, false>) kernel_lightning_indexer_t;

template [[host_name("kernel_lightning_indexer_f32")]]        kernel kernel_lightning_indexer_t kernel_lightning_indexer<float4x4, 1, dequantize_f32, false>;
template [[host_name("kernel_lightning_indexer_f16")]]        kernel kernel_lightning_indexer_t kernel_lightning_indexer<half4x4,  1, dequantize_f16, false>;
template [[host_name("kernel_lightning_indexer_f16_direct")]] kernel kernel_lightning_indexer_t kernel_lightning_indexer<half4x4,  1, dequantize_f16, true>;

#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_lightning_indexer_bf16")]] kernel kernel_lightning_indexer_t kernel_lightning_indexer<bfloat4x4, 1, dequantize_bf16, false>;
#endif

template [[host_name("kernel_lightning_indexer_q4_0")]] kernel kernel_lightning_indexer_t kernel_lightning_indexer<block_q4_0, 2, dequantize_q4_0, false>;
template [[host_name("kernel_lightning_indexer_q4_1")]] kernel kernel_lightning_indexer_t kernel_lightning_indexer<block_q4_1, 2, dequantize_q4_1, false>;
template [[host_name("kernel_lightning_indexer_q5_0")]] kernel kernel_lightning_indexer_t kernel_lightning_indexer<block_q5_0, 2, dequantize_q5_0, false>;
template [[host_name("kernel_lightning_indexer_q5_1")]] kernel kernel_lightning_indexer_t kernel_lightning_indexer<block_q5_1, 2, dequantize_q5_1, false>;
template [[host_name("kernel_lightning_indexer_q8_0")]] kernel kernel_lightning_indexer_t kernel_lightning_indexer<block_q8_0, 2, dequantize_q8_0, false>;
