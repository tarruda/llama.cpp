# Routed expert streaming

This private branch supports bounded routed expert streaming for DeepSeek V4.1 text inference on POSIX systems. Metal shared buffers on Apple silicon receive positioned SSD reads directly. Other supported buffers use staged uploads. Always-used weights remain resident; Engram/PLE embedding rows use the existing lazy loader.

## Options

| Option | Meaning |
| --- | --- |
| `-smoe`, `--stream-moe` | Enable streaming and force `--load-mode none --lazy-mode on`. |
| `--moe-cache MiB` | Total routed cache allocation, including pins. Zero selects an automatic budget. Other model and runtime allocations are additional. |
| `--moe-profile FNAME` | Read a version 1 routing JSON from `llama-imatrix`. |
| `--moe-pin-count N` | Global number of expert bundles selected for permanent encoder pins and phase-specific frequency pins. A positive value requires a profile. |
| `--moe-pin-encoder` | Permanently pin every encoder expert, even if the pin count is smaller. These experts consume the pin count first. |
| `--moe-read-threads N` | Number of concurrent positioned-read workers; default 4. |

An expert bundle contains its gate, up, and down projections. The budget is measured in bytes so it also works with mixed quantizations. Each layer has a bounded cache with enough room for its largest phase-specific pin set plus a dynamic slot, unless that layer is fully resident. Remaining budget is distributed across layers. Unpinned experts use LRU eviction. The cache may use slightly less than the requested budget because complete bundles must fit.

Pin rankings are global within each execution phase. Prefill and decode rankings can allocate different pin counts to a layer; capacity must accommodate both. Overlapping pins retain their data across a phase change. Newly selected pins fill on first use and then remain protected in that phase. Encoder pins remain protected in every phase. If a phase is missing from the JSON, calibration counts are used with a warning. A profile with incompatible architecture, dimensions, missing expert IDs, or invalid counts is rejected.

`--moe-pin-encoder --moe-pin-count 8000` on a 40-layer, 384-expert V4.1 model pins all 7,680 encoder experts plus the 320 highest-ranked decoder experts in each phase. Without an additional decoder pin count, the encoder flag can be used alone. The full MXFP4 encoder exceeds 128 GiB by itself and cannot fit on a 128 GiB device; reserve this option for smaller quantizations.

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

On a 128 GiB M1 Ultra with Flash MXFP4, a 97 GiB routed cache, 2000 pins and 2048-token microbatches reached approximately 109.4 GiB process RSS. Four read workers performed similarly to eight and better than two. Increasing the microbatch to 8192 improved a 5075-token prompt from 51.1 to 64.0 tokens/s at a 90 GiB cache budget, with identical responses and about 1.1 GiB more peak RSS. A smaller routed cache leaves room for these larger batches and saved conversation state. Larger pin sets helped the long prompt but slowed a repeated short chat; select pins for the expected workload.

With a 128000-token context and 8192-token microbatches, the persistent V4.1 state allocation is approximately 699 MiB, including rollback and the encoder frontier. The global KV/index part is approximately 109 MiB. Compute buffers and lazily touched PLE rows are additional memory; the compact context size does not describe total process memory.

Context checkpoints are separate from `--cache-ram`: the server defaults to retaining up to 32 per slot. A checkpoint in this configuration can occupy about 590 MiB. `--ctx-checkpoints 2` bounds their memory cost while retaining recent rollback points; revisiting older branches of a conversation can require more prompt processing.

A 94 GiB routed cache with 2000 pins, four readers, 8192-token microbatches and two checkpoints passed a 20215-token recall request plus a follow-up that reused 20220 cached tokens. Peak process RSS was 109.5 GiB without added swap. Conversation recall, reasoning output, tool calls, tool-result handling and streaming responses also passed. These measurements cover the tested prompts; additional context and different PLE row access patterns can change memory use.

Single-slot text serving defaults to CED prefill. `--prefill-mode full` selects full causal prefill. CED runs the encoder over the full prompt and replays the decoder's final window; every generation step still runs both halves. Streaming does not evict all encoder data at the start of generation.

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

The quantizer preserves V4.1's FP8 Engram byte tables and scales, plus its elementwise BF16 Engram key/query weights. These tensors cannot use the ordinary matrix quantization path. Routed projections and the Engram KV projection remain eligible for quantization.

The low-level API exposes `llama_set_moe_phase`, `llama_get_moe_cache_stats`, and `llama_model_tensor_source`. Callers should set the phase before each prefill or decode operation. The server, imatrix profiler, and benchmark do this automatically. Streamed graphs sharing a model are serialized to protect cache storage while Metal work is in flight.

Row/tensor splitting, MTP loading, Windows streaming, and model loading without GGUF source files are currently rejected. Automatic CPU repacking is disabled for streaming. Use native CPU or Metal buffers for routed tensors. The cache handles quantization strides directly; it does not requantize weights.

Metal attention can decode visible cache rows into temporary F32 working storage and reuse them across query heads. Single-token attention expands only selected rows, computes each query/key score once, and splits independent output channels across threadgroups. Persistent context remains packed. The working allocation is included in the compute buffer, so include it when setting the routed cache budget. `GGML_METAL_DSV41_ATTN_UNPACK_DISABLE=1` selects the original packed attention path for comparisons. Streamed expert groups preserve the original Metal matrix/vector dispatch choice, since those kernels can use different dequantization precision for types such as Q8_0. LoRA tensors targeting nonresident routed weights are rejected; their canonical source descriptors do not expose a resident weight buffer.
