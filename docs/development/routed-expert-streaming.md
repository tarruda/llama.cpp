# Routed expert streaming

This private branch supports bounded routed expert streaming for DeepSeek V4.1 inference on POSIX systems. Metal shared buffers on Apple silicon receive positioned SSD reads directly. Other supported buffers use staged uploads. Always-used weights remain resident; Engram/PLE embedding rows use the existing lazy loader.

## Options

| Option | Meaning |
| --- | --- |
| `-smoe`, `--stream-moe` | Enable streaming and force `--load-mode none --lazy-mode on`. |
| `--moe-cache MiB` | Total routed cache allocation, including pins. Zero selects an automatic budget. Other model and runtime allocations are additional. |
| `--moe-cache-policy lru\|adaptive` | Eviction policy, default `lru`. Adaptive learns live expert usage and excludes `--moe-profile` and positive `--moe-pin-count`. |
| `--moe-profile FNAME` | Read a version 1 routing JSON from `llama-imatrix`. |
| `--moe-pin-count N` | Global number of expert bundles selected for permanent encoder pins and phase-specific frequency pins. A positive value requires a profile. |
| `--moe-pin-encoder` | Permanently pin every encoder expert, even if the pin count is smaller. These experts consume the pin count first. |
| `--moe-read-threads N` | Number of concurrent positioned-read workers; default 4. |

An expert bundle contains its gate, up, and down projections. The budget is measured in bytes so it also works with mixed quantizations. Each layer has a bounded cache with enough room for its largest phase-specific pin set plus a dynamic slot, unless that layer is fully resident. Remaining budget is distributed across layers. Unpinned experts use LRU eviction by default. The cache may use slightly less than the requested budget because complete bundles must fit.

`--moe-cache-policy adaptive` learns decayed routing frequency from hits and misses, breaking frequency ties by recency. Calibration, prefill, and decode have separate histories. During prompt work, part of the resident decode working set stays protected while the remaining slots serve the prompt. Adaptive mode requires streaming and rejects profile input or a positive pin count. Explicit full-encoder residency works in either policy. Learned frequencies last for the loaded model's lifetime; adaptive startup has no calibration history.

For small route sets on Metal shared buffers, selected expert reads start before the shared expert runs. A readiness boundary waits for the reads before routed projections consume the weights. Large route sets and staged-upload buffers retain the synchronous path. `LLAMA_MOE_OVERLAP=0` disables overlap for comparison. Read wait measures the exposed wait; load intervals also include time while the GPU does other work.

For local diagnosis, `LLAMA_MOE_TRACE=/path/to/routes.jsonl` records cache layout and ordered routes at streamed layer boundaries. Fully resident layers do not emit route records. This is separate from imatrix profile output and can add logging overhead; use it to compare cache policies on the same traffic.

Pin rankings are global within each execution phase. Prefill and decode rankings can allocate different pin counts to a layer; capacity must accommodate both. Overlapping pins retain their data across a phase change. Newly selected pins fill on first use and then remain protected in that phase. Encoder pins remain protected in every phase. If a phase is missing from the JSON, calibration counts are used with a warning. A profile with incompatible architecture, dimensions, missing expert IDs, or invalid counts is rejected.

`--moe-pin-encoder --moe-pin-count 8000` on a 40-layer, 384-expert V4.1 model pins all 7,680 encoder experts plus the 320 highest-ranked decoder experts in each phase. Without an additional decoder pin count, the encoder flag can be used alone. The full MXFP4 encoder exceeds 128 GiB by itself and cannot fit on a 128 GiB device; reserve this option for smaller quantizations.

Tensor storage for the Flash model's complete encoder expert bank:

| Routed type | Encoder expert bank | With original resident weights | With Q8_0 default resident weights |
| --- | --- | --- | --- |
| MXFP4 | 134.47 GiB | 150.01 GiB | 142.75 GiB |
| IQ3_XXS | 96.90 GiB | 112.43 GiB | 105.18 GiB |
| Q2_K | 83.06 GiB | 98.59 GiB | 91.34 GiB |

The Q8_0 default reduces other resident tensor storage from 15.53 to 8.28 GiB; the native quantizer preserves norms, router gates, and elementwise Engram weights. These sizes come from actual tensor shapes, quantization block sizes, and a native dry run, not a full-encoder runtime measurement. Decoder expert cache, context, compute buffers, and any resident draft are additional allocations; Engram value and scale tables remain lazy. The complete decoder expert bank has the same storage size. Cache capacities per layer are fixed at model load; phase changes update pin selection, not capacity. Full encoder pinning remains active during generation, which still executes both networks in CED.

## Serving

```bash
./build/bin/llama-server -m model.gguf -ngl 99 -fa on \
    --stream-moe --moe-cache 92160 \
    --moe-profile experts.json --moe-pin-count 2000 \
    --fit off -b 8192 -ub 8192 -c 128000 -np 1 \
    --ctx-checkpoints 2 --cache-ram 512 --no-warmup \
    --host 127.0.0.1 --port 8080
```

The numbers above are starting points, not universal recommendations. Increase the cache only while measuring total physical memory and swap, including after a long prompt. Compare several held-out chats with and without pins: a corpus-specific pin set can reduce the space available for experts needed by a different workload. The server reports read bytes, cache misses, and read wait separately for prefill and decode. Cache lookups are counted at a layer or projection boundary depending on the execution path; compare bytes and time when comparing paths.

Before the dense FP8-to-BF16 upgrade, a 97 GiB routed cache, 2000 pins and 2048-token microbatches reached approximately 109.4 GiB process RSS on a 128 GiB M1 Ultra. Four read workers performed similarly to eight and better than two. Increasing the microbatch to 8192 improved a 5075-token prompt from 51.1 to 64.0 tokens/s at a 90 GiB cache budget, with identical responses and about 1.1 GiB more peak RSS. These historical budgets need to account for the larger BF16 dense weights. Larger pin sets helped the long prompt but slowed a repeated short chat; select pins for the expected workload.

With a 128000-token context and 8192-token microbatches, the persistent V4.1 state allocation is approximately 699 MiB, including rollback and the encoder frontier. The global KV/index part is approximately 109 MiB. Compute buffers and lazily touched PLE rows are additional memory; the compact context size does not describe total process memory.

Context checkpoints are separate from `--cache-ram`: the server defaults to retaining up to 32 per slot. A checkpoint in this configuration can occupy about 590 MiB. `--ctx-checkpoints 2` bounds their memory cost while retaining recent rollback points; revisiting older branches of a conversation can require more prompt processing.

Before the dense FP8-to-BF16 upgrade, a 94 GiB routed cache with 2000 pins, four readers, 8192-token microbatches and two checkpoints passed a 20215-token recall request plus a follow-up that reused 20220 cached tokens. Peak process RSS was 109.5 GiB without added swap. Conversation recall, reasoning output, tool calls, tool-result handling and streaming responses also passed. These measurements cover the tested prompts; additional context and different PLE row access patterns can change memory use.

Single-slot text serving defaults to CED prefill. `--prefill-mode full` selects full causal prefill. CED runs the encoder over the full prompt and replays the decoder's final window; every generation step still runs both halves. Streaming does not evict all encoder data at the start of generation.

## Resident DSpark and images

A converted V4.1 DSpark model can stay resident alongside the streamed target. The draft does not inherit target streaming or profile options. It shares the target token embedding and output projection, and captures only the target attention window from layers 37, 38, and 39. CED prompt replay supplies the same bounded feature window.

```bash
./build/bin/llama-server -m model.gguf -md dspark.gguf \
    --spec-type draft-dspark --spec-draft-n-max 2 --spec-draft-p-min 0 \
    --stream-moe --moe-cache-policy adaptive --moe-cache 81920 \
    -ngl 99 --spec-draft-ngl 99 -fa on --fit off \
    -b 8192 -ub 8192 -c 128000 -np 1 --ctx-checkpoints 2 --no-warmup
```

The draft always evaluates its trained five-position noise block; `--spec-draft-n-max` limits how many proposals the target verifies. Confidence can truncate that prefix further. Target verification uses full causal inference and the target sampler. Rejected suffixes are removed from both contexts. Drafting can cost more than it saves when acceptance is low or its resident weights displace useful cached experts. Compare end-to-end chat timings at the same total memory budget.

Image input uses the V4.1 `deepseek41v` projector, supplied with `--mmproj`. Its image rows stay in reading order and have no V4 alignment padding. Image embeddings use the vision routing bias and bypass Engram; subsequent text n-grams stop at the image boundary. Image serving currently uses full prefill and can also use resident DSpark. Use `--prefill-mode full` or the automatic mode when loading a projector.

Live server measurements with BF16 dense weights on the M1 Ultra:

| Workload | Adaptive streaming | Adaptive streaming with resident DSpark |
| --- | --- | --- |
| Three-turn coding conversation, generation | 4.22, 4.31, 4.44 tokens/s | 4.83, 5.57, 4.83 tokens/s |
| Repeated short explanation, generation | 10.68 tokens/s | 9.59 tokens/s with two proposals; 4.88 with five |
| Image description and OCR | Not measured | 3.96 tokens/s for the logo; 4.55 for the carrots |

The coding comparison used greedy decoding, 88 GiB of cache without drafting and 80 GiB with drafting, a 256 Ki-token context, 8192-token batches, CED, and two checkpoints. DSpark used up to five proposals with confidence threshold 0.5, accepting 74-83% of proposals. Each answer was capped at 192 tokens and hit that cap; these are bounded generation measurements, not completed-answer latency comparisons. Peak RSS was 106.3 GiB without drafting and 105.3 GiB with drafting. Responses with and without speculation differed in these tests. Different target batch shapes remain a possible numerical cause; the rollback fixture reproduced serial continuation logits exactly for all 72 tested cases. Sampling at a positive temperature can change draft acceptance and speed.

The image tests used the BF16 projector, 80 GiB of cache, two proposals without confidence filtering, and full prefill. The 460-token logo prompt took 34.4 seconds to process; the 459-token carrot prompt took 28.3 seconds. Peak RSS was 107.8 GiB. These tests do not establish a universal DSpark speedup.

## Calibration and validation

Use the current branch's [imatrix tool](../../tools/imatrix/README.md#routed-expert-profiles) to obtain both quantization importance data and phase-specific routing counts. The collector sees original expert IDs and canonical source tensor dimensions even when the execution tensor is a smaller cache. Routing counts are not importance scores, and they should not be used to prune experts.

```bash
./build/bin/llama-imatrix -m model.gguf -f calibration.txt \
    --stream-moe --moe-cache 81920 -ngl 99 -fa on --fit off \
    -c 8192 -b 8192 -ub 8192 --process-output --output-frequency 1 \
    --moe-profile-output experts.json --moe-profile-generate 256 \
    --moe-profile-prompts 16 --moe-profile-prompt-tokens 1024 \
    -o imatrix.gguf --seed 41
```

Calibration batches need more working memory than single-token generation, so use a smaller routed cache. The generation pass collects routing statistics without changing the importance matrix. Keep both final files: the quantizer consumes `imatrix.gguf`, while streaming consumes `experts.json`. A zero-count expert has no measured importance; the native quantizer substitutes unit importance for that expert.

The quantizer preserves V4.1's FP8 Engram tables unless a tensor-type override names them. For example, `--tensor-type 'engram_embd\.weight=iq4_nl' --allow-requantize` decodes each FP8 block with its E8M0 scale before quantizing it. Conversion uses the existing bounded row slabs. Runtime lazy lookup dequantizes only the requested rows and does not read the original scales for quantized tables. The scale tensors remain in the GGUF, unchanged, for layout compatibility. Elementwise BF16 Engram key/query weights remain unchanged; routed projections and the Engram KV projection are independently eligible for quantization.

Perplexity and reference-logit generation use full causal inference. For V4.1, set `-c 512 -b 512 -ub 512` to evaluate one sequence at a time; a larger batch than context makes the perplexity tool create multiple sequences, which this runtime does not support. An 80 GiB routed cache leaves room for the evaluation buffers. Keep Metal residency enabled on this device. Reference `.kld` files generated with `--save-all-logits` can later be supplied to `--kl-divergence-base` when evaluating a quantized model.

The low-level API exposes `llama_set_moe_phase`, `llama_get_moe_cache_stats`, and `llama_model_tensor_source`. Callers should set the phase before each prefill or decode operation. The server, imatrix profiler, and benchmark do this automatically. Streamed graphs sharing a model are serialized to protect cache storage while Metal work is in flight.

Row/tensor splitting, MTP loading, Windows streaming, and model loading without GGUF source files are currently rejected. Automatic CPU repacking is disabled for streaming. Use native CPU or Metal buffers for routed tensors. The cache handles quantization strides directly; it does not requantize weights.

Metal attention can decode visible cache rows into temporary F32 working storage and reuse them across query heads. Single-token attention expands only selected rows, computes each query/key score once, and splits independent output channels across threadgroups. Persistent context remains packed. The working allocation is included in the compute buffer, so include it when setting the routed cache budget. `GGML_METAL_DSV41_ATTN_UNPACK_DISABLE=1` selects the original packed attention path for comparisons. Streamed expert groups preserve the original Metal matrix/vector dispatch choice, since those kernels can use different dequantization precision for types such as Q8_0. LoRA tensors targeting nonresident routed weights are rejected; their canonical source descriptors do not expose a resident weight buffer.
