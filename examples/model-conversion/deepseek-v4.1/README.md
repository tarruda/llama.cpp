DeepSeek V4.1 conversion and numerical reference

The production converter accepts the released `DeepseekV41ForCausalLM` checkpoint. The backbone export excludes the separate DSpark and vision tensors. Dense FP8 weights use the checkpoint's 32 x 32 scale blocks and are stored as BF16 by default. `--fp8-as-q8` explicitly opts into lossy Q8_0 storage for those weights. The V4 converter retains its existing 128 x 128 behavior.

Routed MXFP4 experts are repacked without changing values, one expert at a time. Engram values and E8M0 scales are stored separately as raw I8 tensors and written in bounded row chunks. Neither the expert collection nor the large Engram tables need to become fully resident during conversion. The converter writes the Engram hash constants and installs this branch's V4.1 text template.

```sh
python convert_hf_to_gguf.py /path/to/DeepSeek-V4.1-Flash \
    --outtype bf16 --split-max-size 105G --no-tensor-first-split \
    --outfile DeepSeek-V4.1-Flash-MXFP4.gguf
python convert_hf_to_gguf.py /path/to/DeepSeek-V4.1-Flash \
    --dspark --outtype bf16 --outfile DeepSeek-V4.1-Flash-DSpark-MXFP4.gguf
python convert_hf_to_gguf.py /path/to/DeepSeek-V4.1-Flash \
    --mmproj --outtype bf16 --outfile DeepSeek-V4.1-Flash-mmproj-BF16.gguf
```

DSpark exports three stages with 128 experts each, the target projection, Markov head and F32 confidence projection. Target embeddings and the output matrix are shared rather than duplicated. Its `dflash.target_model_architecture = deepseek41` metadata distinguishes V4.1's carried hyper-connections and target attention input features from the V4 draft. Export does not yet enable speculative inference: the runtime rejects this draft until the V4.1 draft graph is implemented. Vision export likewise does not enable a V4.1 vision runtime.

The following tools validate the numerical implementation with small synthetic weights.

`reference.py` imports the released `inference/model.py` and `engram.py` without changing them. `cpu_kernel.py` supplies CPU implementations of the six imported TileLang kernel functions. This allows the official Python layer, routing, RoPE, compression, shared-cache, and Engram logic to run on a small CPU fixture. It is not CUDA instruction emulation: reduction order and elementary-function approximations can differ from the GPU kernels.

The fixture keeps all 40 backbone layers and the released KV/index source topology. Widths, expert counts, Engram tables, sliding window, and index/candidate limits are reduced. Its default 33-token prompt crosses the 16-token window, ratio-2 compression boundaries, top-8 indexing threshold, and four candidate blocks of four positions. Four subsequent decode steps exercise incomplete and completed compressor groups. Weights are deterministic and nonuniform; all residual copies, output groups, and routed experts have distinct parameters. The synthetic tokenizer has 128 unique entries and exercises the official hash implementation, but does not replace real-tokenizer validation.

For text chat with a GGUF carrying an older embedded template, override it with `--jinja --chat-template-file models/templates/deepseek-ai-DeepSeek-V4.1.jinja`. The override preserves reminders, applies the official reasoning-history policy and joins typed text blocks with two newlines. It accepts reasoning effort `low`, `high`, `max`, or an integer from 1 to 100; decimal strings are also accepted for CLI settings. Thinking mode defaults to effort 75. Native DeepSeek preprocessing orders tool results, and the parser recognizes V4.1's spaced DSML tags. Reminder text stays in the model prompt and is excluded from parsed assistant output and grammar prefill. This template is for the native text chat path; it does not implement the official encoder's vision or auxiliary task interfaces.

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

Use `compare-stages.py` with an existing native evaluation-callback capture to locate the first observed numerical difference. It reads the native `index.jsonl` records (`position`, `name`, `file`, `type`, `shape`, `strides`) and raw F32/I32 tensor files, and compares named forward stages against the reference `stages-NNNNN.npz` files. No model execution or tensor injection occurs. Both captures must use the same weights, tokens, full-forward batching, state and reference corrections. The reader expects matching token spans, not decoder-only CED replay layouts.

```sh
.venv/bin/python examples/model-conversion/deepseek-v4.1/compare-stages.py \
    /tmp/dsv41-reference-fp8 /tmp/dsv41-native \
    --output /tmp/dsv41-drift.json --dump-first /tmp/dsv41-first-drift.npz
```

The terminal summary reports the first unequal value, the first value exceeding `atol + rtol * abs(reference)`, the first unequal attention/FFN residual, and the first changed routed-expert set. Each numerical event includes the call position, layer, stage, absolute token position, full reference-layout element index and both values. The JSON includes every matched stage, error counts, relative L2 error, missing stages and unmapped reference stages. Integer routing IDs always require exact agreement; expert order is retained, with set changes reported separately. Use `--position 0` to isolate prefill or `--stage-regex '^dsv41_(attn_post|layer)-'` to focus on residual outputs. Defaults are `--atol 1e-6 --rtol 1e-5`; use zero for exact numerical equality. The optional NPZ saves the first failing pair and the preceding matched stage, not all graph inputs. Nonfinite values and missing mapped stages fail the comparison. Exit code 0 means only that matched stages meet the requested tolerance; it does not accept unmapped operations or cache/state handling.

Stage order follows the reference archive's capture order, not the native scheduler order or token order across the whole model. The first observed difference brackets where to investigate; it does not prove which instruction caused it. Reference captures with the extra `dsv41_*` HC tensors provide finer coverage than module hooks alone. The existing CPU check script also tests threshold reporting, token coordinates, strided reads, routing comparisons and missing-stage handling.

The released standalone `generate.py` calls `Transformer.forward`, which processes every supplied prompt token through all 40 layers. It does not implement the decoder-skipping and bounded-replay optimization described in the technical report.

`ced-reference.py` evaluates the decoder bounded-replay schedule described in technical report sections 2.2 and 3.2.2. It runs all prompt tokens through encoder layers 0-19, carries the four residual copies and final FFN pre coefficients, and projects layer 20's global KV/index keys from the normalized encoder frontier. Only the last window of encoder outputs runs through decoder layers 20-39. During replay, SWA cannot read before the replay start; global context retains its normal causal visibility and is not overwritten. Subsequent decode tokens traverse all 40 layers.

```sh
.venv/bin/python examples/model-conversion/deepseek-v4.1/ced-reference.py \
    --source /Users/thiago/ml-models/huggingface/deepseek-ai/DeepSeek-V4.1-Flash/inference \
    --output /tmp/dsv41-ced-reference --prompt-length 193 --stable-index
```

The reduced fixture's replay window is 16 tokens. `--replay-tokens` can select a smaller window or cover the complete prompt for an equivalence check. `--stable-index` records an explicit lower-ID tie policy for context selection; router selection is unchanged. The output records per-layer token counts and includes the encoder frontier, normalized global-cache input, and later state. The reference retains the whole frontier for inspection. Bounded replay is intentionally approximate, so its logits are not expected to equal full forward once prompt tokens are omitted from decoder replay. Short prompts and complete-prompt replay match the existing CPU full-forward reference exactly in the tested cases.

The native `llama_prefill()` API selects CED for DeepSeek-V4.1. A prompt batch can request no outputs or only its final token's logits. Encoder-only chunks publish global context and retain BF16 residual copies plus F32 pre coefficients in the history ring. The final output request gathers the last window and replays decoder layers 20-39. Ordinary `llama_decode()` runs all layers; use it for calibration and verification. Other architectures retain their existing decode path. `llama-server` and the embedded server in `llama-cli` default to `--prefill-mode auto`, which selects CED for supported configurations. Use `--prefill-mode full` for the full-forward comparison path, or `--prefill-mode ced` to require a supported configuration. CED currently requires `-np 1`, text generation, and no speculative decoding; auto uses full processing otherwise. Prompt chunks and cached prompt suffixes use `llama_prefill()` even when they contain one token. Generation continues through `llama_decode()`.

Snapshots include the frontier, replay boundary and decoder readiness. Partial checkpoints omit the append-only global KV/index payloads and require the matching live token prefix when restored. The runtime reports its sliding window and retained history boundary so the server can restore a checkpoint or reprocess a prefix whose history was overwritten. Snapshot version `d5410003` distinguishes full and partial payloads. Generation is rejected while prefill is incomplete. A failed CED replay requires another prefill or snapshot restore because it can overwrite earlier decoder rows. At 128000 context and microbatch 512, the real model's frontier tensors add 25.0098 MiB, bringing persistent cache/frontier tensors to 154.0430 MiB before allocation alignment, token history and graph scratch. Frontier capacity depends on the window and microbatch, not prompt length.

Reduced native checks cover 1-, 15-, 16-, 17-, 33- and 193-token prompts. Short prompts through one window match full-forward logits/caches exactly. With 193-token prompts and microbatches 64 and 7, CED retains identical encoder stages and all published global caches, then runs 16 tokens through each decoder layer. Chunked prefill, pending-state restore, appended turns, generation and supported rollback are checked on CPU and shared/private Metal state. These checks validate scheduling and state handling; reference logit drift and real-model quality remain open.

Initial scratch reservation covers full-forward processing, encoder-only chunks, and decoder replay of the entire window, including microbatches smaller than that window. Reservation uses a memory view without changing live tokens or cache state. Reduced lifecycle checks at microbatches 1, 7 and 64 stay within the initial buffers; CPU, shared/private Metal and Metal with CPU caches have matching allocated and no-allocation estimates. Two shared accounting fixes count duplicate buffer types once and preserve explicit backend assignments during size estimation; neither changes model arithmetic.

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

`DSV41_HC_SPLIT` computes the HC affine transforms, sigmoid gates, and Sinkhorn normalization. It adds epsilon after the row or column sum, as the released equations specify. The CPU and Metal implementations use ordered four-element sums; Metal uses precise division and the compensated `dsv41_exp` helper. A captured `precise::exp` error of one F32 ULP changed a post gate and crossed a BF16 residual boundary. The helper corrects that case without changing other models' HC paths.

The CPU reference adapter now evaluates the initial HC softmax as separate maximum subtraction, exponential, sum, and division steps, following the released kernel. Older references used PyTorch's fused softmax, which can round differently. Keep those outputs as a separate reference variant and check the recorded adapter hash. Changing only this reference evaluation changed the reduced fixtures' full logits by several percent with identical weights; agreement with either CPU variant alone does not establish CUDA parity.

```sh
build-dsv41/bin/test-backend-ops test -o DSV41_HC_SPLIT -b MTL0
```

These cases cover one and 20 normalization iterations, single-token and multi-token inputs, strided rows, and the captured post-gate rounding boundary.

CPU reference arithmetic can also cause consequential differences. Separate diagnostics evaluate HC and attention exponentials and HC inverse square roots in F64, then round each result to F32 before subsequent arithmetic. Under this common contract, both 33-token reduced fixtures with four decode steps match native full logits to about `1e-7` relative L2 and have exact decoded cache values. Original reference outputs remain separate. This does not establish CUDA parity, bitwise equality of every intermediate, or real-model quality; F32 reductions and matrix/vector dispatch can still change rounding.

`DSV41_SWIGLU` fuses gate/up clamping, SiLU, optional route weighting, and BF16 rounding. Gate and up inputs contain BF16 values in F32; route weights remain F32 and are applied before the result is rounded. The shared expert omits route weights. Metal uses precise exponential and division functions and preserves the multiplication order. Quantization for the down projection remains a separate operation.

```sh
build-dsv41/bin/test-backend-ops test -o DSV41_SWIGLU -b MTL0
```

These cases cover clamped and unclamped activations, optional weights, tail elements, the real expert width of 2304, and strided rows across all four tensor dimensions.

`DSV41_SET_ROWS` packs F32 source rows directly into an I8 cache using the activation scale and rounding rules for MXFP8, MXFP4, or NVFP4. The result aliases the cache. Nonnegative destination IDs must be distinct within a call; an ID of -1 skips that source row. Source rows, IDs, and cache rows can have outer strides. CPU and Metal implementations preserve unmapped rows and padding. The model uses these writes as explicit dependencies of its cache readers. Incomplete ratio-2 groups use skipped IDs.

```sh
build-dsv41/bin/test-backend-ops test -o DSV41_SET_ROWS -b MTL0
```

These cases cover all three packed formats, single-token and multi-token writes, skipped rows, and contiguous or strided views.

`DSV41_INDEX_SCORES` reads packed MXFP4 index keys without expanding the key cache. Queries are F32 matrices of dequantized MXFP4 values, arranged as `[dimension, heads, tokens]`. Head weights contain BF16 values in F32. The operation rounds the dot products, weighted head terms, and final head sum to BF16 values in F32. It masks future positions with negative infinity. Without candidates, its output has one score per key and token; with candidates, it has one score per position in the candidate blocks. Neither layout has a temporary head dimension.

```sh
build-dsv41/bin/test-backend-ops test -o DSV41_INDEX_SCORES -b MTL0
```

These cases cover causal masks, empty candidate selections, strided inputs, ratio-1 and ratio-2 caches, and the real index width of 128 with 32 heads. With KV offload enabled, index keys for Metal layers are allocated on Metal and both their writes and scoring run there. Disabling KV offload keeps keys and scoring on the CPU. Snapshot I/O also supports private Metal buffers.

`DSV41_SELECT` selects context positions and optional candidate blocks from F32 index scores. It masks future positions, sets the newest visible candidate block's score to positive infinity, and breaks score ties by lower original ID. When consuming compact scores, pass the scorer's candidate tensor so selection maps slots back to original context positions. Candidate block IDs must be sorted, unique and followed by any -1 padding. Both output sections are sorted by ID and padded with -1. Metal finds the score threshold with radix histograms, then emits selected IDs in order. Integer comparisons preserve subnormal scores and treat signed zeros as equal.

```sh
build-dsv41/bin/test-backend-ops test -o DSV41_SELECT -b MTL0
```

These cases cover exact ties, negative infinity, signed zeros, subnormal scores, strided inputs, and 128000 keys. Selection runs with the scorer on the cache's backend. With KV offload enabled on Metal, score matrices and selected IDs stay on Metal. Disabling KV offload keeps scoring and selection on the CPU.

`DSV41_ATTN` reads an MXFP8 window ring and optional NVFP4 context rows directly. Queries and output contain BF16 values in F32. It uses 64-entry tiles, F32 online softmax, BF16 probability rounding before the value product, and a denominator-only attention sink. CPU and Metal preserve ordered dot products and sums. The Metal kernel supports head widths up to 512; its caches follow layer placement and KV offload settings. Snapshot I/O supports shared and private Metal caches.

`ggml_dsv41_attn_set_window_start` adds a lower bound to SWA positions for bounded replay. Its default is zero, and it does not restrict global context. CPU attention evaluates exponentials in F64 and rounds them to F32 before accumulation. A captured `expf` error of one F32 step changed a softmax correction and crossed a BF16 output boundary; the F64 evaluation agrees with the independently checked nearest-F32 values in that case.

```sh
build-dsv41/bin/test-backend-ops test -o DSV41_ATTN -b MTL0
```

These cases cover prefill, early decode, ring wraparound, ratio-1 and ratio-2 context, duplicate and masked IDs, strided inputs, and the real 64-head, 512-wide shape. Metal uses a local exponential with high/low F32 range reduction and a degree-12 Taylor polynomial. The standard Metal `precise::exp` changed a captured denominator by one F32 step and crossed a BF16 output boundary; the local implementation fixes that case. The `dsv41_exp` helper is used by V4.1 attention, compressor pooling and HC split; SwiGLU retains its existing exponential.

`DSV41_POOL` reads F32 compressor values and scores from history rings and pools completed ratio-2 groups with ordered F32 softmax weights. It returns BF16 values in F32 and zero for incomplete groups. The graph writes both histories with the existing `SET_ROWS` operation before pooling. History allocation and pooling follow cache placement, including private Metal buffers and CPU placement when KV offload is disabled. The text graph no longer uses CPU custom callbacks.

```sh
build-dsv41/bin/test-backend-ops test -o DSV41_POOL -b MTL0
```

These cases cover single-token and multi-token inputs, strided histories and positions, ring wraparound, incomplete groups, and the real width of 512. Snapshot layout is unchanged; replay checks cover partial groups and suffix rollback.

Longer private fixtures use a 193-token prompt and nine decode tokens, crossing the 80-row and 23-row history rings multiple times. Snapshot restore, reset, and rollback with consistent batching preserve logits and live cache bytes. Changing between matrix and vector evaluation can change HC projection rounding: at one captured point, identical inputs and RMS factors produce different BF16 residuals after HC mixing. The short fixture's batching agreement therefore does not establish general batching parity. Neither fixture replaces real-model quality validation.

Indexers without candidates still use the full cache width: for a 128000-key cache and 512-token microbatch, one F32 score matrix is 250 MiB. Candidate-consuming indexers use compact scores; 16384 candidate positions at the same microbatch size require 32 MiB. These are individual tensor sizes, not peak scratch measurements. Selection requires no CPU copy of the score matrix. Full-width source-indexer scratch, attention matrix tiling and shared KV reuse across heads remain performance work.

The native indexer breaks exact score ties by ascending original position or block ID. This keeps masked future padding from changing the selected context. PyTorch `topk` can choose other equally scoring IDs. Check cutoff scores and membership when comparing index selections, or record an explicit reference variant with the same tie rule; do not silently replace the original reference outputs.
