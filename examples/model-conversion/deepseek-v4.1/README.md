DeepSeek V4.1 numerical reference

`reference.py` imports the released `inference/model.py` and `engram.py` without changing them. `cpu_kernel.py` supplies CPU implementations of the six imported TileLang kernel functions. This allows the official Python layer, routing, RoPE, compression, shared-cache, and Engram logic to run on a small CPU fixture. It is not CUDA instruction emulation: reduction order and elementary-function approximations can differ from the GPU kernels.

The fixture keeps all 40 backbone layers and the released KV/index source topology. Widths, expert counts, Engram tables, sliding window, and index/candidate limits are reduced. Its default 33-token prompt crosses the 16-token window, ratio-2 compression boundaries, top-8 indexing threshold, and four candidate blocks of four positions. Four subsequent decode steps exercise incomplete and completed compressor groups. Weights are deterministic and nonuniform; all residual copies, output groups, and routed experts have distinct parameters. The synthetic tokenizer has 128 unique entries and exercises the official hash implementation, but does not replace real-tokenizer validation.

From the repository root, using an environment with PyTorch 2.11, NumPy, SymPy, tokenizers, and Pillow:

```sh
.venv/bin/python examples/model-conversion/deepseek-v4.1/check-cpu-kernel.py
.venv/bin/python examples/model-conversion/deepseek-v4.1/reference.py \
    --source /Users/thiago/ml-models/huggingface/deepseek-ai/DeepSeek-V4.1-Flash/inference \
    --output /tmp/dsv41-reference-fp8
.venv/bin/python examples/model-conversion/deepseek-v4.1/reference.py \
    --source /Users/thiago/ml-models/huggingface/deepseek-ai/DeepSeek-V4.1-Flash/inference \
    --output /tmp/dsv41-reference-bf16 --dense-type bf16 --expert-type bf16
```

`weights.npz` and `weights.json` preserve parameter storage bytes, names, shapes, and dtypes. `reference.json` records configuration, seeds, source/adapter hashes, software versions, and invocation results. Each `stages-NNNNN.npz` contains input token IDs, complete final-position logits, original routing IDs/weights, intermediate activations, HC residual/pre coefficients, and live cache/state values for that call. Floating-point stage values are exported as F32, including values rounded to BF16 by the model; parameter storage is kept separately in its original format. Files are emitted into the explicit output directory, not this source directory.

The CPU checks cover FP4 midpoint ties, adjacent source nibble packing, zero scales, the activation scale rule, FP8 rounding, block-scaled GEMMs, attention sinks/masks/duplicate indices, and HC normalization. The reference runner checks finite stages, valid distinct original expert IDs, and normalized routing weights. These are foundations for native graph comparisons; they do not establish correctness of the rebuilt llama.cpp model or fidelity to an actual execution of the released CUDA kernels.

The released `Indexer.forward` binds `shared_attn.index_k` only when an owning layer has a new compressed latent. On an incomplete ratio-2 decode group, the pointer can still refer to the final decoder source from the previous call. For example, after a 33-token prefill and one decode token, layer 2 at position 34 reads layer 20's index cache. Its declared source is layer 2. `--bind-index-cache-owner` installs a forward hook that binds each owner's cache before its indexer runs, including incomplete groups. This correction is optional, recorded in `reference.json` under `errata`, and leaves the official files unchanged. Keep uncorrected and corrected outputs in separate directories.

Export a reference fixture for native graph comparisons:

```sh
.venv/bin/python examples/model-conversion/deepseek-v4.1/export-fixture.py \
    /tmp/dsv41-reference-fp8 /tmp/dsv41-reference-fp8/model.gguf
```

The exporter preserves BF16/F32 parameters and raw Engram bytes. By default, it decodes FP8/FP4 weights to F32 without changing their values and stacks routed experts in original ID order. Add `--keep-expert-fp4` to retain routed weights as MXFP4 and exercise native expert kernels; the exporter checks every packed matrix against the decoded source values. Activation precision is recorded separately from weight storage. The exporter checks the official source hashes recorded by the reference runner and consumes every exported parameter; it is only for the small fixtures.

After building GGML, compare the native packed activation rows and graph operation with the Python definitions:

```sh
.venv/bin/python examples/model-conversion/deepseek-v4.1/check-native-quant.py \
    --library build-dsv41/bin/libggml-base.dylib --backend cpu
.venv/bin/python examples/model-conversion/deepseek-v4.1/check-native-quant.py \
    --library build-dsv41/bin/libggml-base.dylib --backend metal
build-dsv41/bin/test-backend-ops test -o DSV41_ACT_QUANT -b MTL0
```

Adjust the library path and backend device name to the build. Omitting `--backend` checks only packed CPU rows. The script checks exact packed bytes and decoded values for MXFP4, NVFP4, and MXFP8, plus BF16 graph rounding. Its inputs include FP4/FP8 midpoint neighbors, E4M3 scale ties, signed zero, subnormal values, and random BF16 values across 40 exponents. The backend tests add contiguous and strided tensor layouts. These activation quantizers are separate from the existing weight quantizers; `DSV41_ACT_QUANT` returns F32 values and does not allocate packed cache storage.

`DSV41_ENGRAM` fuses the normalized weighted dot product, signed square root, sigmoid, and residual addition in F32. It preserves the positive gate offset at a zero dot product; the graph rounds the result to BF16 separately. `DSV41_ROPE` uses shared interleaved cosine/sine inputs and rounds its result to BF16 values in F32. The graph builds those inputs once per microbatch for the window, compressed context, and ratio-2 group positions. This avoids repeating trigonometry for every head and layer.

```sh
build-dsv41/bin/test-backend-ops test -o DSV41_ENGRAM,DSV41_ROPE -b MTL0
```

These backend cases cover real model widths, inverse rotations, and strided inputs. They compare CPU and Metal implementations; they do not establish end-to-end model parity.

`DSV41_HC_SPLIT` computes the HC affine transforms, sigmoid gates, and Sinkhorn normalization. It adds epsilon after the row or column sum, as the released equations specify. The CPU and Metal implementations use ordered four-element sums; Metal uses precise division and exponential functions. Small coefficient differences can cross BF16 rounding boundaries in the residual streams and accumulate through later layers.

```sh
build-dsv41/bin/test-backend-ops test -o DSV41_HC_SPLIT -b MTL0
```

These cases cover one and 20 normalization iterations, single-token and multi-token inputs, and strided rows.

`DSV41_SWIGLU` fuses gate/up clamping, SiLU, optional route weighting, and BF16 rounding. Gate and up inputs contain BF16 values in F32; route weights remain F32 and are applied before the result is rounded. The shared expert omits route weights. Metal uses precise exponential and division functions and preserves the multiplication order. Quantization for the down projection remains a separate operation.

```sh
build-dsv41/bin/test-backend-ops test -o DSV41_SWIGLU -b MTL0
```

These cases cover clamped and unclamped activations, optional weights, tail elements, the real expert width of 2304, and strided rows across all four tensor dimensions.

`DSV41_SET_ROWS` packs F32 source rows directly into an I8 cache using the activation scale and rounding rules for MXFP8, MXFP4, or NVFP4. The result aliases the cache. Nonnegative destination IDs must be distinct within a call; an ID of -1 skips that source row. Source rows, IDs, and cache rows can have outer strides. CPU and Metal implementations preserve unmapped rows and padding. This primitive is not yet connected to the model graph, which still uses CPU cache callbacks.

```sh
build-dsv41/bin/test-backend-ops test -o DSV41_SET_ROWS -b MTL0
```

These cases cover all three packed formats, single-token and multi-token writes, skipped rows, and contiguous or strided views.

The native indexer breaks exact score ties by ascending original position or block ID. This keeps masked future padding from changing the selected context. PyTorch `topk` can choose other equally scoring IDs. Check cutoff scores and membership when comparing index selections, or record an explicit reference variant with the same tie rule; do not silently replace the original reference outputs.
