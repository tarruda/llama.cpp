# Implementation notes: Metal KV attention optimization

## Purpose

This is a technical history of the following six commits:

```text
1eb2c68c353f802b2cb8d7ebf205c78454e43671
metal: pack q8_0 loads for 256-wide KV

966ab5920a95c160b2cabadfbd1a398ecf1e93b4
metal: materialize q8_0 KV for high-GQA decode

c0c6c3b3b44033cf07f916de563c0a352f0df4a2
metal: materialize q8_0 KV for deep prompts

4ad9aaac2d117aae84130d98141c5d2c88d3fb45
metal: materialize q8_0 KV for 128-wide GQA

dfe574140be9d010089f26ed0c2503ad762a98a8
metal: optimize f16 attention for 128-wide GQA

924461601dcc13593c2890968c9bc38a452826fa
metal: reuse f16 scheduling for q8_0 KV
```

It records the implementation, dispatch policy, memory layout, benchmarks, correctness coverage, and limitations. The first four commits optimize Q8_0 KV attention. The fifth commit optimizes the native F16 kernels used as their performance reference. The sixth commit lets eligible Q8_0 materialization reuse the same F16 scheduling policies.

## Target workload

The work targets Metal flash attention with Q8_0 K and V caches on an Apple M1 Ultra with 128 GiB unified memory.

The first benchmark model is Nex N2 Pro, based on Qwen 397B. Its relevant attention shape is:

```text
query head size:       256
key head size:         256
value head size:       256
query heads:            32
KV heads:                2
GQA ratio:               16
query batch for decode:   1
query batch for prompt: 512
```

The GQA ratio means that 16 independent query heads share each physical K/V head.

The fourth commit extends the same implementation to Hy3. Its relevant attention shape is:

```text
query head size:       128
key head size:         128
value head size:       128
query heads:            64
KV heads:                8
GQA ratio:                8
query batch for decode:   1
query batch for prompt: 512
```

Hy3 uses conventional GQA rather than the 256-wide attention shape used by Nex. It was initially excluded from materialization solely because the dispatch gate required 256-element query, key, and value heads.

The fifth commit targets the same Hy3 geometry with a native F16 KV cache. Hy3 uses full attention, so each layer must process the entire visible cache. Unlike sliding-window attention, its cache scan is not bounded to a recent window. Unlike gated recurrent layers, it cannot replace that scan with fixed-size state. The linear context-length cost is therefore fundamental, but the amount of redundant work and the way that work is distributed across the GPU are still kernel design choices.

The sixth commit applies those scheduling choices after Q8_0 K/V has been materialized into F16 scratch. It does not change the persistent Q8_0 representation or the materialization gate.

## Original problem

The persistent Q8_0 cache was much smaller than F16, but Q8_0 token generation degraded rapidly with context length:

| Existing context | Original Q8_0 generation | Native F16 generation |
| --- | ---: | ---: |
| 0 | 22.82 t/s | 23.88 t/s |
| 10K | 13.18 t/s | 22.32 t/s |
| 20K | 9.49 t/s | 21.02 t/s |

Prompt processing also fell behind at depth:

| Existing context | Original Q8_0 prompt | Native F16 prompt |
| --- | ---: | ---: |
| 0 | 177.80 t/s | 180.12 t/s |
| 10K | 140.11 t/s | 154.75 t/s |
| 20K | 115.44 t/s | 139.82 t/s |

Q8_0 stores 32 values in one 34-byte block:

```c
struct block_q8_0 {
    ggml_half d;
    int8_t qs[32];
};
```

Every value is reconstructed as:

```text
value = float(qs[i]) * float(d)
```

The old attention kernels performed that reconstruction while scanning the cache. In Nex decode, the same unique KV head was dequantized independently for 16 query heads. In a 512-query prompt, each query head was further divided into 64 eight-query tiles, causing much greater repeated conversion of the deep prefix.

The optimization progressed in three stages:

1. Make one Q8_0 vector load cheaper.
2. Remove repeated conversion from high-GQA decode by materializing F16 scratch.
3. Reuse the same materialization for deep prompt processing.

## Commit 1: packed Q8_0 vector loads

### Commit

```text
metal: pack q8_0 loads for 256-wide KV
```

### Files

```text
ggml/src/ggml-metal/ggml-metal.metal | 10 lines changed
tests/test-backend-ops.cpp           |  6 lines added
```

### Shader change

The commit added `dequantize_q8_0_t4_packed`:

```metal
template <typename type4>
void dequantize_q8_0_t4_packed(
        device const block_q8_0 * xb,
        short il,
        thread type4 & reg) {
    device const packed_char4 * qs =
        (device const packed_char4 *) xb->qs;
    const float d = xb->d;

    reg = (type4) (float4(qs[il]) * d);
}
```

The previous generic helper loaded four bytes through scalar array indexing. The packed helper treats the byte payload as an array of four-byte signed vectors, converts one vector to `float4`, and applies the block scale.

Only the `DK = DV = 256` Q8_0 vector flash-attention specialization switched to the packed helper:

```text
kernel_flash_attn_ext_vec_q8_0_dk256_dv256
```

Other Q8_0 head-size specializations retained the generic helper. Applying the packed helper broadly regressed an existing 64-wide case, so the change was deliberately specialized to the measured Nex shape.

### Tests added

Focused Nex cases were added to `test-backend-ops`:

- 256-wide K and V
- 32 query heads represented by 2 KV heads and repeat ratio 16
- one-query decode
- native and permuted cache layouts
- short unaligned context for correctness
- 10K and 20K contexts for performance

### Results

Exact attention operation:

| Context | Scalar load | Packed load | Improvement |
| --- | ---: | ---: | ---: |
| 10K | 1714.86 us | 1682.99 us | 1.9% |
| 20K | 3917.08 us | 3412.05 us | 12.9% |

End-to-end generation:

| Existing context | Scalar load | Packed load | Improvement |
| --- | ---: | ---: | ---: |
| 0 | 22.82 t/s | 22.91 t/s | 0.4% |
| 10K | 13.18 t/s | 13.90 t/s | 5.5% |
| 20K | 9.49 t/s | 10.19 t/s | 7.4% |

The packed load reduced the cost of one conversion but did not change how often the cache was converted. The context-length slope therefore remained poor.

## Commit 2: high-GQA decode materialization

### Commit

```text
metal: materialize q8_0 KV for high-GQA decode
```

### Files

```text
ggml/src/ggml-metal/ggml-metal-device.cpp | pipeline creation
ggml/src/ggml-metal/ggml-metal-device.h   | pipeline declarations
ggml/src/ggml-metal/ggml-metal-impl.h     | shared kernel arguments
ggml/src/ggml-metal/ggml-metal-ops.cpp    | selection and dispatch
ggml/src/ggml-metal/ggml-metal-ops.h      | scratch sizing API
ggml/src/ggml-metal/ggml-metal.cpp        | allocation accounting
ggml/src/ggml-metal/ggml-metal.metal      | kernels and GQA support
tests/test-backend-ops.cpp                | correctness and perf cases
```

The commit contains 460 insertions and 213 deletions across eight files.

### Intermediate two-head GQA specialization

Before introducing scratch, the vector flash-attention template was generalized to process more than one query head per threadgroup.

The template gained a query-head count parameter `H`, with supported values 1 and 2. Per-head state became arrays indexed by `H`:

- query vectors in threadgroup memory
- online-softmax sum `S`
- online-softmax maximum `M`
- score fragments
- value/output accumulators
- partial-result storage

The threadgroup maps `H` adjacent query heads to one shared KV head. K and V are loaded and dequantized once, then applied to two independent query-head states.

The host added:

```text
kernel_flash_attn_ext_vec_gqa2_q8_0_dk256_dv256
```

The GQA2 path was restricted to compatible Q8_0 decode shapes and allocated threadgroup memory proportional to `H`.

Results against the packed one-head path:

| Context | One head | Two heads | Improvement |
| --- | ---: | ---: | ---: |
| 10K | 1672.29 us | 1153.30 us | 31.0% |
| 20K | 3409.13 us | 2154.12 us | 36.8% |

End-to-end generation reached 16.70 t/s at 10K and 12.49 t/s at 20K.

A four-head version was tested but not retained. Four independent query, softmax, and output states increased register pressure enough to reduce occupancy and make the kernel slower. Explicitly moving Q8_0 scale multiplication outside selected K and V operations was also tested and removed because it was slower.

### Fundamental decode issue

Two-head sharing reduced 16 direct Q8_0 traversals per unique KV head to eight, but the Q8_0 attention slope remained about 7.2 times the F16 slope.

The exact native F16 operation was only about 218 us at 10K and 357 us at 20K. This showed that the main remaining cost was repeatedly consuming Q8_0, not the attention math common to both paths.

### Q8_0-to-F16 conversion kernel

The commit added:

```text
kernel_flash_attn_ext_q8_0_to_f16
```

One GPU thread converts one 32-value Q8_0 block. The dispatch covers K and V in one one-dimensional grid. The first half of the grid converts K and the second half converts V.

For each block, the kernel:

1. Decodes the linear block index into block, position, KV head, and stream coordinates.
2. Computes the source byte offset with the original tensor strides.
3. Loads the F16 scale.
4. Loads eight `packed_char4` vectors.
5. Converts and scales them.
6. Stores eight contiguous `half4` vectors.

The destination is physically contiguous in logical flash-attention order:

```text
[head element, cache position, KV head, stream]
```

The resulting F16 byte strides are:

```text
nb0 = 2
nb1 = 2 * head_size
nb2 = nb1 * context_length
nb3 = nb2 * kv_head_count
```

The kernel therefore performs dequantization and layout conversion in the same pass.

### Shared kernel arguments

`ggml_metal_kargs_flash_attn_ext_q8_0_to_f16` was added to `ggml-metal-impl.h` so the C++ host and Metal shader share an ABI-compatible argument structure.

It contains:

- K dimensions
- K byte strides
- V byte strides
- total block count

The host verifies that twice the block count fits signed 32-bit dispatch accounting.

### Pipeline creation

`ggml_metal_library_get_pipeline_flash_attn_ext_q8_0_to_f16` was added to compile and cache the conversion pipeline.

The vector attention pipeline helper gained two effective-execution parameters:

- `nhptg`: query heads per threadgroup
- `use_f16_kv`: select F16 cache type and F16 row-stride constants even though the original operation tensors remain Q8_0

This is necessary because the Metal specialization name and function constants must match the actual scratch representation, not the persistent tensor metadata.

### Scratch allocation

The Metal operation allocation calculation gained:

```text
ggml_metal_op_flash_attn_ext_extra_q8_f16
```

The operation-local layout became:

```text
flash-attention output
padding storage
mask-block storage
vector partial-result storage
F16 K scratch
F16 V scratch
```

The runtime reconstructs offsets in exactly the same order. K and V scratch sizes are padded to 16-byte boundaries.

The F16 scratch size is:

```text
2 * NKV * NHKV * (DK + DV) * streams
```

For Nex at 20K:

```text
2 * 20000 * 2 * (256 + 256)
= 40960000 bytes
= 39.0625 MiB
```

The persistent cache remains Q8_0. The F16 expansion exists only as operation-local scratch.

### Padding and temporary regions

The existing final-chunk padding kernel copies raw bytes. It can pad F16 scratch, but only when the host supplies F16 row strides and reserves enough F16-sized padding storage.

The scratch eligibility helper is used during allocation as well as dispatch, ensuring that:

- the padding region uses effective F16 row sizes
- K scratch and V scratch do not overlap padding, block-mask, or reduction storage
- the last partial 32-position vector chunk remains valid

### Synchronization

After conversion, the host calls:

```text
ggml_metal_op_concurrency_reset
```

This emits an encoder memory barrier before padding or attention reads the new F16 buffers. Existing barriers remain between padding and attention, and between vector attention and its partial-result reduction.

### Decode selection policy

The initial materialization path required:

```text
K type = Q8_0
V type = Q8_0
NQ = 1
DK = 256
DV = 256
NKV >= 1024
GQA ratio >= 8
matching K/V position, head, and stream dimensions
```

Short contexts and unsupported shapes retain direct Q8_0 attention. The two-head specialization remains available as a fallback for eligible lower-GQA shapes.

### Decode results

Exact operation:

| Context | Packed Q8_0 | Two-head Q8_0 | F16 scratch | Native F16 |
| --- | ---: | ---: | ---: | ---: |
| 10K | 1672.29 us | 1153.30 us | 303.71 us | 217.89 us |
| 20K | 3409.13 us | 2154.12 us | 481.16 us | 356.57 us |

End-to-end generation:

| Existing context | Original Q8_0 | F16 scratch | Native F16 | Scratch/F16 |
| --- | ---: | ---: | ---: | ---: |
| 0 | 22.82 t/s | 22.92 t/s | 23.88 t/s | 96.0% |
| 10K | 13.18 t/s | 21.45 t/s | 22.32 t/s | 96.1% |
| 20K | 9.49 t/s | 19.85 t/s | 21.02 t/s | 94.4% |

From 10K to 20K, optimized Q8_0 retains 92.5 percent of its generation speed while F16 retains 94.2 percent. The previous roughly 75 percent retention was removed.

## Commit 3: deep prompt materialization

### Commit

```text
metal: materialize q8_0 KV for deep prompts
```

### Files

```text
ggml/src/ggml-metal/ggml-metal-device.cpp | non-vector F16 selection
ggml/src/ggml-metal/ggml-metal-device.h   | updated helper signature
ggml/src/ggml-metal/ggml-metal-ops.cpp    | shared conversion dispatch
tests/test-backend-ops.cpp                | prompt correctness and perf
```

The commit contains 108 insertions and 93 deletions across four files. Most deletions and insertions in `ggml-metal-ops.cpp` move the already tested conversion setup out of the vector-only branch so both attention implementations can consume it.

### Prompt reuse opportunity

The non-vector Metal attention kernel handles eight query tokens per threadgroup.

For a 512-query prompt:

```text
512 / 8 = 64 query tiles per query head
```

Nex has 16 query heads per KV head, so one unique deep-prefix block can be dequantized across approximately:

```text
64 tiles * 16 query heads = 1024 tile/head combinations
```

The existing F16 non-vector kernel avoids Q8_0 conversion and uses its optimized F16 matrix path. A one-time conversion pass is negligible relative to the repeated Q8_0 work.

### Eligibility extension

The scratch helper no longer requires the vector path. It now recognizes two query-size classes:

```text
decode: NQ = 1
prompt: NQ >= 64
```

Both continue to require:

```text
Q8_0 K and V
DK = DV = 256
NKV >= 1024
GQA ratio >= 8
matching K/V dimensions
```

Query batches from 2 through 63 retain the previous direct path. Depth-zero `pp512` also remains direct because its visible cache is below the 1024-position threshold.

### Shared conversion setup

The commit moved these operations before the vector/non-vector branch:

- scratch eligibility evaluation
- effective K and V buffer selection
- effective K and V strides
- conversion argument construction
- conversion dispatch
- conversion-to-consumer memory barrier

After that common setup, either attention branch sees:

```text
bid_k, bid_v
nb10_attn through nb13_attn
nb20_attn through nb23_attn
use_q8_f16
```

When scratch is disabled, those values describe the original K/V tensors. When enabled, they describe the contiguous F16 buffers.

### Non-vector pipeline selection

`ggml_metal_library_get_pipeline_flash_attn_ext` gained `use_f16_kv`, matching the vector helper's effective-type behavior.

When true, it selects:

```text
kernel_flash_attn_ext_f16_dk256_dv256
```

and compiles function constants with:

```text
ns10 = 256
ns20 = 256
```

instead of Q8_0 block strides.

The non-vector host path now passes effective buffers and strides to:

- final-chunk padding
- flash-attention arguments
- flash-attention buffer bindings

The quantized shared-memory staging term is disabled when F16 scratch is active because the selected F16 kernel does not need the Q8_0 dequantization staging region.

### Prompt tests

The correctness suite gained a Nex-shaped prompt case with:

- `DK = DV = 256`
- GQA ratio 16
- `NKV = 1025`
- `NQ = 64`
- mask enabled
- permuted cache layout
- unaligned final cache chunk

Performance cases were added for Q8_0 and F16 at:

```text
NQ = 512
NKV = 10000 and 20000
```

### Prompt results

Exact operation:

| Context | Direct Q8_0 | F16 scratch | Native F16 | Direct/scratch speedup |
| --- | ---: | ---: | ---: | ---: |
| 10K | 48.41 ms | 28.59 ms | 28.34 ms | 1.69x |
| 20K | 97.08 ms | 57.32 ms | 56.91 ms | 1.69x |

End-to-end prompt processing:

| Existing context | Original Q8_0 | F16 scratch | Native F16 | Scratch/F16 |
| --- | ---: | ---: | ---: | ---: |
| 0 | 177.80 t/s | 179.17 t/s | 180.12 t/s | 99.5% |
| 10K | 140.11 t/s | 152.01 t/s | 154.75 t/s | 98.2% |
| 20K | 115.44 t/s | 136.33 t/s | 139.82 t/s | 97.5% |

The scratch gate is inactive at depth zero. The small difference there is normal benchmark variation. At depth, the change improves Q8_0 prompt processing by 8.5 percent at 10K and 18.1 percent at 20K.

### Decode regression check

Moving conversion setup out of the vector branch did not regress decode. Final exact operation measurements were approximately:

```text
10K: 304 us
20K: 484 us
```

These match the measurements before the prompt extension within normal run-to-run variance.

## Commit 4: 128-wide GQA materialization

### Commit

```text
metal: materialize q8_0 KV for 128-wide GQA
```

### Files

```text
ggml/src/ggml-metal/ggml-metal-ops.cpp | eligibility extension
tests/test-backend-ops.cpp             | Hy3 correctness and perf cases
```

The commit contains 14 insertions and one deletion across two files. No new shader or pipeline was required because the conversion kernel is dimension-generic and Metal already provides 128-wide F16 vector and non-vector flash-attention specializations.

### Root cause

Hy3 has 64 query heads, eight KV heads, and 128 elements per query, key, and value head. Its GQA ratio is eight, so it satisfies the reuse threshold used by the Nex materialization path.

The original eligibility test nevertheless required all three head sizes to equal 256:

```cpp
op->src[0]->ne[0] == 256 &&
op->src[1]->ne[0] == 256 &&
op->src[2]->ne[0] == 256
```

As a result, Hy3 always used direct Q8_0 attention. Every query head or prompt tile repeatedly loaded and dequantized the persistent cache. The large context-length degradation was therefore not caused by a separate classic-attention implementation. It was caused by a valid high-reuse shape missing the existing fast-path gate.

### Eligibility extension

The head-size check now accepts 128 or 256 elements and requires K and V to match the query head size:

```cpp
const bool use_head_size =
    op->src[0]->ne[0] == 128 || op->src[0]->ne[0] == 256;

return use_head_size &&
    op->src[1]->ne[0] == op->src[0]->ne[0] &&
    op->src[2]->ne[0] == op->src[0]->ne[0] &&
    /* existing depth, query-count, shape, and GQA checks */;
```

All other requirements remain unchanged. In particular, materialization still requires Q8_0 K and V, at least 1024 visible KV positions, either one query or at least 64 queries, matching K/V dimensions, and a GQA ratio of at least eight.

The existing converter derives its block count from the runtime head size. The contiguous F16 strides are also derived from the tensor dimensions. Selecting a 128-wide F16 pipeline therefore required no changes outside the eligibility check.

### Tests added

Correctness coverage uses the exact Hy3 head geometry:

- `DK = DV = 128`
- eight physical KV heads
- GQA repeat ratio eight, producing 64 query heads
- `NKV = 1024`
- `NQ = 1` for vector decode
- `NQ = 64` for non-vector prompt processing

Performance coverage compares Q8_0 scratch and native F16 at 10K and 20K for both one-query decode and 512-query prompt processing.

### Exact operation results

| Operation | Context | Q8_0 with F16 scratch | Native F16 | Q8_0 overhead |
| --- | ---: | ---: | ---: | ---: |
| Decode | 10K | 336.50 us | 209.96 us | 60.3% |
| Decode | 20K | 587.63 us | 351.64 us | 67.1% |
| Prompt | 10K | 24.61 ms | 24.18 ms | 1.8% |
| Prompt | 20K | 49.24 ms | 48.20 ms | 2.2% |

Prompt processing reaches practical parity because 512 queries provide enough work to amortize the conversion. Decode retains the cost of converting the visible cache once per layer invocation, so it remains slower than a cache already stored as F16.

### End-to-end Hy3 results

The model benchmark uses a 295.03B-parameter Hy3 IQ3_XXS model on the M1 Ultra.

| Existing context | Original Q8_0 prompt | Optimized Q8_0 prompt | Native F16 prompt |
| --- | ---: | ---: | ---: |
| 0 | 157.62 t/s | 157.26 t/s | 161.87 t/s |
| 10K | 66.72 t/s | 100.24 t/s | 100.15 t/s |
| 20K | not measured | 72.55 t/s | 71.07 t/s |

| Existing context | Original Q8_0 generation | Optimized Q8_0 generation | Native F16 generation |
| --- | ---: | ---: | ---: |
| 0 | 20.03 t/s | 19.89 t/s | 22.11 t/s |
| 10K | 10.16 t/s | 13.54 t/s | 16.78 t/s |
| 20K | not measured | 10.37 t/s | 13.75 t/s |

At 10K, prompt processing improves by 50.2 percent and matches F16 within benchmark variation. Generation improves by 33.3 percent but remains 19.3 percent behind F16. At 20K, prompt processing also matches F16, while generation remains 24.6 percent behind.

The depth-zero gate is inactive because the visible cache contains fewer than 1024 positions. The small depth-zero changes are benchmark variation rather than an effect of the new path.

### Rejected follow-up experiments

Several more aggressive changes were measured and removed:

- Assigning one thread to both corresponding K and V blocks reduced parallelism. Decode regressed to 408.87 us at 10K and 730.70 us at 20K.
- Reshaping the converter dispatch into a multidimensional grid to remove dynamic index divisions produced 337.32 us and 586.61 us, effectively no improvement over the simpler linear dispatch.
- A direct packed-Q8_0 two-query-head kernel took 675.88 us and 1209.42 us.
- A direct four-query-head kernel took 1055.55 us and 1686.06 us. Additional per-head query, softmax, score, and output state increased register pressure enough to overwhelm the saved cache loads.

The retained implementation is therefore the smallest change and the fastest tested path for Hy3. Further generation gains require reducing the one-time Q8_0-to-F16 conversion cost or avoiding full-cache conversion without reintroducing repeated dequantization.

## Commit 5: native F16 128-wide GQA optimization

### Commit

```text
metal: optimize f16 attention for 128-wide GQA
```

### Files

```text
ggml/src/ggml-metal/ggml-metal-device.cpp | pipeline selection
ggml/src/ggml-metal/ggml-metal-device.h   | updated helper signature
ggml/src/ggml-metal/ggml-metal-ops.cpp    | dispatch and temporary sizing
ggml/src/ggml-metal/ggml-metal.metal      | prompt kernel and decode reducer
tests/test-backend-ops.cpp                | Hy3-shaped F16 correctness cases
```

The commit contains 107 insertions and 45 deletions across five files.

### Remaining F16 costs

Native F16 attention does not pay a dequantization cost, but it still performs two context-length-dependent operations for every query:

```text
scores = Q * transpose(K)
output = softmax(scores) * V
```

The existing matrix kernel processed eight adjacent queries per threadgroup. Every group loaded a K matrix and a V matrix, used each for one 8-row query matrix, and then discarded it. The next eight queries repeated the same K/V loads.

The vector kernel divided one-query decode across 32 partial workgroups. It increased each workgroup from one to as many as four SIMD groups as the context grew. For 128-wide F16 attention this put more work and shared state in each threadgroup instead of exposing more independent workgroups to the GPU. At 16K and deeper, 32 partials also left less parallelism available than the device could use.

These are scheduling and reuse limitations, not changes to the full-attention algorithm. The optimized path still reads and computes over the complete context.

### Sixteen-query prompt tile

The matrix flash-attention template now derives the number of 8-row query matrices from its query tile size:

```text
NQM = Q / 8
```

The direct F16 matrix branches maintain one score matrix and one output matrix per 8-row query matrix. For the new `Q = 16` specialization, each K matrix is loaded once and multiplied by two Q matrices. Each V matrix is also loaded once and applied to two probability matrices. The online-softmax state remains independent for every query row.

The new pipeline is:

```text
kernel_flash_attn_ext_q16_f16_dk128_dv128
```

The existing `Q = 8` specializations remain unchanged for other shapes. Host dispatch selects the 16-query tile only when all of the following hold:

```text
K type = F16
V type = F16
query head size = 128
value head size = 128
query count >= 64
GQA ratio >= 8
```

At this commit, the Q8_0 materialization path continued to select the existing query tile because its operation tensors were Q8_0 even though the temporary consumer buffers were F16. Commit 6 removes that source-metadata restriction for eligible materialized operations.

### F16 decode work distribution

For one-query, 128-wide, high-GQA F16 attention, the vector path now caps each attention workgroup at two SIMD groups instead of allowing four. This reduces per-threadgroup state and exposes more schedulable work at the measured Hy3 shape.

At 16K or more visible KV positions, the path also increases the number of partial attention workgroups from 32 to 64. Each partial independently computes online-softmax state and an output vector for a slice of the cache. The final reducer combines those partials with the normal numerically stable maximum and exponential rescaling.

The old reducer assumed one SIMD lane per partial workgroup, so it could only directly represent 32 partials. Launching 64 SIMD groups would also require 2048 threads, beyond Metal's 1024-thread threadgroup limit. The generalized reducer instead makes each lane consume partials in strides of 32:

```text
partial index = lane, lane + 32, ...
```

It first finds the global maximum, then combines the softmax sums and output vectors after rescaling each partial to that maximum. The reducer launches at most 32 SIMD groups, or 1024 threads, while accepting either 32 or 64 producer workgroups.

The 64-workgroup gate requires:

```text
K type = F16
V type = F16
query head size = 128
value head size = 128
query count = 1
visible KV positions >= 16384
GQA ratio >= 8
```

Other vector attention shapes retain 32 workgroups and their previous SIMD-group limit.

### Exact operation results

The original and final Hy3-shaped F16 flash-attention measurements were:

| Operation | Context | Original | Optimized | Runtime reduction |
| --- | ---: | ---: | ---: | ---: |
| Prompt, 512 queries | 10K | 24.025 ms | 22.688 ms | 5.6% |
| Prompt, 512 queries | 20K | 47.822 ms | 44.984 ms | 5.9% |
| Decode, one query | 10K | 210.64 us | 187.69 us | 10.9% |
| Decode, one query | 20K | 354.50 us | 336.98 us | 4.9% |

The 10K decode case benefits from the two-SIMD-group cap. The 20K case also uses 64 partial workgroups and the generalized reducer. Doubling the producer count improves GPU occupancy, but it adds more partial output traffic and reduction work, so its net gain is smaller.

### End-to-end Hy3 results

The optimized native F16 benchmark was compared with the previously measured F16 baseline:

| Existing context | Operation | Original F16 | Optimized F16 | Change |
| --- | --- | ---: | ---: | ---: |
| 0 | Prompt | 161.87 t/s | 162.02 t/s | +0.1% |
| 0 | Generation | 22.11 t/s | 21.97 t/s | -0.6% |
| 10K | Prompt | 100.15 t/s | 104.59 t/s | +4.4% |
| 10K | Generation | 16.78 t/s | 17.11 t/s | +2.0% |
| 20K | Prompt | 71.07 t/s | 76.85 t/s | +8.1% |
| 20K | Generation | 13.75 t/s | 13.91 t/s | +1.2% |

The depth-zero results are effectively unchanged. The larger prompt gain at depth shows that K/V reuse matters increasingly as the full-attention cache scan grows. Generation improves more modestly end to end because attention is only part of each token's total model work and the 64-way split adds reduction overhead.

### Correctness coverage

Two exact Hy3-shaped F16 cases were added:

- a 64-query, 1024-position case that selects the 16-query prompt tile
- a one-query, 16384-position case that selects the 64-workgroup decode path

Both focused cases passed existing flash-attention numerical tolerances. A full unfiltered Metal backend suite was started accidentally and manually stopped after making steady progress without observed failures. That run did not complete and is not counted as full-suite validation for this commit.

## Commit 6: reuse F16 scheduling for Q8_0 materialization

### Commit

```text
metal: reuse f16 scheduling for q8_0 KV
```

### Files

```text
ggml/src/ggml-metal/ggml-metal-ops.cpp | effective F16 scheduling selection
tests/test-backend-ops.cpp             | materialized prompt and deep-decode cases
```

The commit contains 10 insertions and four deletions across two files. It adds no shader or pipeline specialization. It reuses the kernels introduced by commit 5.

### Missed scheduling reuse

Q8_0 materialization already replaced the attention consumer's buffers and strides with contiguous F16 scratch. Pipeline selection also used an effective F16 K/V type. Three host scheduling decisions nevertheless continued to test the original operation tensor types:

```text
op->src[1]->type == GGML_TYPE_F16
op->src[2]->type == GGML_TYPE_F16
```

Those tests are false for materialized attention because the graph tensors remain Q8_0. Consequently, 128-wide Q8_0 materialization used F16 data and F16 pipelines but missed three scheduling choices from commit 5:

- the 16-query prompt tile
- the two-SIMD-group decode cap
- 64 decode workgroups at 16K or deeper

The conversion work was already shared, but its F16 consumer still used the older Q8_0-selected work decomposition.

### Effective F16 predicate

The host now centralizes the distinction between persistent tensor type and attention consumer type:

```cpp
static bool ggml_metal_op_flash_attn_ext_use_f16_attn(const ggml_tensor * op) {
    return (op->src[1]->type == GGML_TYPE_F16 && op->src[2]->type == GGML_TYPE_F16) ||
        ggml_metal_op_flash_attn_ext_use_q8_f16(op);
}
```

The predicate is true for native F16 K/V and for Q8_0 operations that satisfy the complete materialization gate. It replaces the original physical-type check in the three scheduling decisions. Their existing shape and workload checks remain in place, so the newly shared behavior is restricted to matching 128-wide, GQA-8-or-higher attention at the previously measured query and context thresholds.

This separation is important:

```text
persistent storage type = Q8_0
attention consumer type = F16
scheduling type          = F16
```

Direct Q8_0 attention, short contexts, query counts from 2 through 63, GQA ratios below 8, and unsupported head sizes remain unchanged. The 256-wide materialization path also remains unchanged by these scheduling policies because each downstream policy independently requires 128-wide query and value heads.

### Prompt selection

For an eligible materialized prompt, `use_q16` now sees the effective F16 consumer. A 128-wide prompt with at least 64 queries and GQA ratio at least eight therefore selects:

```text
kernel_flash_attn_ext_q16_f16_dk128_dv128
```

Each threadgroup processes two 8-row query matrices and reuses each K and V matrix across all 16 queries, matching native F16 prompt scheduling.

### Decode selection

For eligible one-query, 128-wide materialized attention, the vector kernel now caps the number of SIMD groups per workgroup at two. At 16384 or more visible KV positions, it also produces 64 partial attention results instead of 32. The generalized reducer from commit 5 combines those partials without exceeding Metal's threadgroup-size limit.

The Q8_0-to-F16 conversion still runs once before attention. This commit only changes how the resulting F16 cache is consumed.

### Correctness coverage

The existing 128-wide Q8_0 prompt case was strengthened to use:

```text
query/K/V head size = 128
KV heads             = 8
GQA ratio             = 8
visible KV positions  = 1025
queries               = 64
layout                 = permuted
```

The unaligned 1025-position tail exercises F16 final-chunk padding while the permuted source layout exercises conversion into contiguous scratch. The 64-query batch selects the 16-query F16 prompt tile.

A separate Q8_0 decode case uses 16385 visible positions, one query, and the same Hy3 head geometry. It jointly exercises materialization, unaligned final-chunk padding, the two-SIMD-group cap, 64 partial workgroups, and generalized reduction.

### Performance status

This commit changes only scheduling selection and does not include a new recorded end-to-end benchmark set. The reused schedules were measured independently with native F16 in commit 5, but Q8_0 still pays the preceding materialization pass. A separate Q8_0 benchmark is required to quantify the net model-level gain rather than inferring it from the native F16 results.

## Q8_0 materialization execution flow

The cumulative path is:

```text
FLASH_ATTN_EXT with Q8_0 K/V
        |
        v
check exact dimensions, context, query count, and GQA ratio
        |
        +-- ineligible --> existing direct attention path
        |
        v
reserve output-adjacent F16 K/V scratch
        |
        v
one Q8_0-to-F16 conversion dispatch for K and V
        |
        v
Metal memory barrier
        |
        v
optional final-chunk padding using F16 strides
        |
        v
optional padding/block-mask barrier
        |
        +-- NQ = 1 ----> F16 vector flash attention
        |                    |
        |                    +-- 128-wide GQA-8 --> at most 2 SIMD groups
        |                    |
        |                    +-- KV >= 16K ------> 64 partial workgroups
        |                    |
        |                    v
        |               partial-result reduction
        |
        +-- NQ >= 64 --> F16 non-vector flash attention
        |                    |
        |                    +-- 128-wide GQA-8 --> 16-query tile
        |
        v
attention output remains in the original operation format
```

The persistent model cache is never changed to F16. Cache insertion, cache sequence operations, attention rotation, and cache lifetime remain unchanged.

## Q8_0 materialization dispatch policy

The current materialization gate is intentionally narrow:

| Property | Required value |
| --- | --- |
| K type | Q8_0 |
| V type | Q8_0 |
| Query/K head size | 128 or 256 |
| V head size | equal to the query/K head size |
| Visible KV positions | at least 1024 |
| Query count | exactly 1 or at least 64 |
| GQA ratio | at least 8 |
| K/V positions | equal |
| K/V head counts | equal |
| K/V stream counts | equal |

All other operations retain existing Metal behavior.

For materialized 128-wide K/V, attention scheduling adds these narrower gates:

| Workload | Additional policy |
| --- | --- |
| At least 64 queries and GQA ratio at least 8 | 16-query prompt tile |
| One query and GQA ratio at least 8 | at most two SIMD groups per attention workgroup |
| One query, GQA ratio at least 8, and at least 16384 KV positions | 64 partial workgroups |

Native F16 and eligible Q8_0-to-F16 materialized attention use the same policies.

## Cumulative changed-file responsibilities

### `ggml/src/ggml-metal/ggml-metal.metal`

- Adds packed four-byte Q8_0 vector dequantization.
- Uses it for the 256-wide direct vector specialization.
- Generalizes vector attention to one or two query heads per threadgroup.
- Adds the two-head Q8_0 GQA specialization.
- Adds dense Q8_0-to-F16 K/V conversion.
- Generalizes the native F16 matrix path to reuse K/V across multiple 8-row query matrices.
- Adds a 16-query specialization for 128-wide F16 GQA.
- Generalizes vector partial reduction to support more than 32 producer workgroups.

### `ggml/src/ggml-metal/ggml-metal-impl.h`

- Defines the conversion kernel's shared host/shader arguments.

### `ggml/src/ggml-metal/ggml-metal-device.cpp` and `.h`

- Compile and cache the conversion pipeline.
- Select one-head or two-head vector specializations.
- Allow vector and non-vector pipeline helpers to select an effective F16 KV type and contiguous F16 stride constants independently of source tensor metadata.
- Select query-tile-specific matrix pipeline names.

### `ggml/src/ggml-metal/ggml-metal-ops.cpp` and `.h`

- Implement exact scratch eligibility.
- Calculate F16 scratch and F16 padding sizes.
- Construct output-adjacent buffer offsets.
- Dispatch conversion and barriers.
- Substitute effective buffers and strides.
- Route decode and prompt processing to the appropriate F16 attention implementation.
- Preserve direct Q8_0 and two-head fallbacks.
- Select the 16-query prompt tile for eligible 128-wide F16 consumers.
- Use two SIMD groups for matching F16-consumer decode and 64 partial workgroups at 16K or deeper.
- Treat eligible Q8_0 materialization as an F16 attention consumer for those scheduling decisions.
- Reserve temporary storage for either 32 or 64 decode partials.

### `ggml/src/ggml-metal/ggml-metal.cpp`

- Includes F16 KV scratch in Metal operation allocation size.

### `tests/test-backend-ops.cpp`

- Adds Nex-shaped correctness cases.
- Adds Hy3-shaped 128-wide decode and prompt correctness cases.
- Covers native and permuted layouts.
- Covers aligned and unaligned cache tails.
- Covers multiple streams, sinks, bias, and softcap for decode scratch.
- Covers the non-vector prompt scratch path.
- Adds exact 10K and 20K Q8_0/F16 performance cases for Nex and Hy3 decode and prompt processing.
- Adds exact native F16 cases for the 16-query prompt tile and 64-workgroup decode path.
- Adds a permuted, unaligned Q8_0 prompt case for the 16-query tile.
- Adds an unaligned 16K Q8_0 decode case for the 64-workgroup path.

## Validation performed

### Build

```sh
cmake --build build --target test-backend-ops llama-bench -j 16
```

### Full Metal flash-attention correctness

After the fourth commit:

```text
4759/4759 tests passed
```

The full matrix includes unrelated F16 and quantized attention cases, providing regression coverage for pipeline-selection changes.

The fifth commit's full unfiltered Metal run was manually stopped before completion. It made steady progress without observed failures, but it does not replace a completed full-suite result.

The sixth commit adds focused Q8_0 cases for both F16 scheduling branches. No completed full-suite result was recorded for that commit.

### Focused correctness

Focused cases exercised:

- one-head short-context fallback
- contiguous decode scratch
- permuted decode scratch
- two streams
- unaligned final chunks
- masks
- attention sinks
- ALiBi bias
- logit softcap
- permuted 64-query non-vector prompt scratch
- 128-wide GQA-8 decode and non-vector prompt scratch
- 128-wide F16 GQA-8 prompt with a 16-query tile
- 128-wide F16 GQA-8 decode with 64 partial workgroups
- permuted 128-wide Q8_0 GQA-8 prompt with an unaligned tail and a 16-query tile
- 128-wide Q8_0 GQA-8 decode with an unaligned 16K tail and 64 partial workgroups

The two focused F16 cases added by the fifth commit passed.

### Diff hygiene

`git diff --check` and staged-diff checks passed before each commit.

## Benchmark commands

The local Python environment was activated as requested:

```sh
source .venv/bin/activate
```

### Decode

```sh
build/bin/llama-bench \
    -m /path/to/Nex-N2-Pro-IQ3_XXS-00001-of-00004.gguf \
    -ngl 999 \
    -fa on \
    -ctk q8_0 \
    -ctv q8_0 \
    -p 0 \
    -n 128 \
    -d 0,10000,20000 \
    -r 3 \
    --progress
```

### Prompt processing

```sh
build/bin/llama-bench \
    -m /path/to/Nex-N2-Pro-IQ3_XXS-00001-of-00004.gguf \
    -ngl 999 \
    -fa on \
    -ctk q8_0 \
    -ctv q8_0 \
    -p 512 \
    -n 0 \
    -d 0,10000,20000 \
    -r 3 \
    --progress
```

Native F16 comparisons use `-ctk f16 -ctv f16` with all other parameters unchanged.

### Operation-level performance

```sh
build/bin/test-backend-ops perf \
    -o FLASH_ATTN_EXT \
    -b MTL0 \
    -p 'hsk=256.*kv=(10000|20000).*type_K=(q8_0|f16),type_V=(q8_0|f16)'
```

The `nb=1` cases select decode. The `nb=512` cases select prompt processing.

### Hy3 end-to-end

```sh
build/bin/llama-bench \
    --mmap 1 \
    -m /path/to/Hy3-IQ3_XXS.gguf \
    -ngl 99 \
    -fa auto \
    -t 1 \
    -d 0,10000,20000 \
    -ctk q8_0 \
    -ctv q8_0
```

### Hy3 operation-level performance

```sh
build/bin/test-backend-ops perf \
    -o FLASH_ATTN_EXT \
    -b MTL0 \
    -p 'hsk=128.*nh=8.*kv=(10000|20000)'
```

## Compatibility and numerical behavior

The conversion reconstructs Q8_0 values and stores them as F16 before attention. The F16 kernel can perform reductions in a different order from direct Q8_0, so byte-identical results are not guaranteed. Results are required to remain within existing flash-attention numerical tolerances.

The changes do not alter:

- Q8_0 quantization format
- persistent KV-cache bytes
- cache writes
- attention rotation
- model graph semantics
- output tensor format
- non-Metal backends
- non-Q8_0 cache types
- unsupported Metal shapes

## Known limitations and remaining work

### Supported shape specialization

Materialization currently targets matching 128-wide or 256-wide query, K, and V heads. Other common head sizes require separate benchmarking before extending the gate.

### Device-specific crossover

The 1024-position, 64-query, and GQA-ratio thresholds were validated on an M1 Ultra. Other Apple GPU generations and smaller memory configurations may have different conversion, bandwidth, and scratch-allocation tradeoffs.

### Scratch peak

One eligible Nex 20K operation requests 39.0625 MiB for F16 K/V. Hy3 has eight 128-wide KV heads and requests 78.125 MiB at the same depth. These allocations are in addition to padding and existing temporary storage. Full model benchmarks completed within the 128 GiB budget, but an allocator or GPU trace is still needed to report exact physical reuse and peak scratch residency across layers.

### Query counts 2 through 63

Small batched or speculative decode continues to use direct attention. A separate small-query kernel or a measured materialization crossover could improve those workloads.

### GQA ratios below 8

The scratch path remains disabled. The two-head specialization can still reduce repeated dequantization for compatible even GQA ratios.

### Other quantized cache types

Q4_0, Q4_1, Q5_0, Q5_1, and other block formats are unchanged. They would need type-specific conversion kernels and independent accuracy/performance evaluation.

### Prompt processing at shallow depth

The scratch path is disabled below 1024 visible positions. This preserves depth-zero performance but leaves any smaller crossover opportunity unexplored.

### Full-attention scaling

Hy3 full attention must consume all visible K/V positions for every attention layer. The F16 scheduling changes reduce the constant cost but do not change that linear relationship. Sliding-window attention and recurrent layers degrade more slowly because they perform less context-dependent work, not because they use a generally faster version of the same full-attention operation.

### F16-consumer specialization

The 16-query prompt tile and 64-workgroup decode policy are restricted to matching 128-wide F16 consumers, at least GQA-8, and the measured query/context thresholds. A consumer can use native F16 K/V or eligible Q8_0 materialized into F16 scratch. Other head sizes, lower GQA ratios, BF16, and direct quantized consumption require independent performance and correctness evaluation before using the same dispatch policy.

## Main conclusion

The large improvement came from changing the number of Q8_0 conversions, not from finding one unusually slow instruction.

Packed loads reduced the cost of each conversion. Two-head GQA sharing reduced the number of conversions by half. Transient F16 materialization reduced conversion to one dense pass per layer invocation and reused the existing optimized F16 attention kernels for all query heads and prompt tiles.

This preserved the persistent memory advantage of Q8_0 while bringing Nex deep-context generation to 94 to 96 percent of F16 and Nex deep prompt processing to 97 to 98 percent of F16 on the target M1 Ultra.

Extending the same path to Hy3 showed that the design applies beyond 256-wide Qwen attention. Hy3 deep prompt processing reached F16 parity at both 10K and 20K. Hy3 generation improved substantially, but the remaining per-layer conversion pass leaves it at about 75 to 81 percent of F16 generation speed at those depths.

The native F16 follow-up found additional avoidable overhead inside full attention itself. Reusing each K/V matrix across 16 prompt queries reduced exact prompt runtime by about 6 percent. Redistributing F16 decode work reduced exact runtime by about 11 percent at 10K and 5 percent at 20K. End-to-end Hy3 gains were smaller because the model still performs all non-attention work and full attention still scales linearly with context length.

The final follow-up removed a metadata mismatch between Q8_0 storage and its materialized F16 consumer. Once conversion has produced contiguous F16 K/V, eligible 128-wide Q8_0 attention now uses the same prompt reuse and decode work distribution as native F16 instead of selecting schedules from the persistent Q8_0 tensor type.
