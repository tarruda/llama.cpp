# Routed expert streaming

This private branch supports routed expert streaming for DeepSeek V4.1 on POSIX systems. Always-used tensors stay resident. The routed expert gate, up, and down projections are loaded from GGUF shards into a bounded cache. Engram/PLE embedding rows use the separate lazy tensor loader.

## Interface

| Option | Meaning |
| --- | --- |
| `-smoe`, `--stream-moe` | Enable routed expert streaming. This forces `--load-mode none --lazy-mode on`. Without `--moe-cache`, the runtime allocates only enough slots for one active expert set. |
| `--moe-cache MiB` | Allocate an adaptive routed expert cache. The value is the total routed cache budget and requires `--stream-moe`. Other model, context, compute, and lazy row allocations are additional. |

The earlier LRU, routing JSON, expert pin count, encoder pin, and read worker options were removed. Adaptive routing is now the only cache policy. Streaming uses four positioned-read workers internally. `llama-imatrix` continues to collect quantization importance data, but it no longer emits or consumes expert frequency JSON.

An expert bundle contains one expert's gate, up, and down projections. Cache sizing uses the exact tensor strides, so it supports MXFP4, Q3_K, Q2_K, and other uniform routed quantizations. A budget that cannot hold one complete bundle is rejected. A zero cache setting still needs transient storage for the six experts selected by one V4.1 token; those slots have no persistent capacity beyond the active working set.

```bash
./build/bin/llama-server -m model.gguf --mmproj mmproj-BF16.gguf \
    --stream-moe --moe-cache 87040 \
    -ngl 99 -fa on --fit off -c 262144 -np 1 \
    -b 8192 -ub 8192 -t 16 -tb 16 \
    --prefill-mode ced --ctx-checkpoints 2 --cache-ram 1024 \
    --host 127.0.0.1 --port 8080
```

## Cache behavior

Partial caches use one global slot pool across all streamed layers. A `(layer, expert)` mapping identifies each resident bundle. Eviction ranks candidates by decayed frequency for the current execution phase and then by recency. Calibration, prefill, and decode have separate frequency histories. The history lasts for the loaded model's lifetime and starts empty.

When a conversation returns from decode to prefill, the cache protects the most useful resident decode experts. It leaves at least 10 percent of dynamic slots, or one layer's 384 experts when that is larger, available for prompt work. A large prompt can release the least useful temporary protections when its active group would otherwise need to be split.

When a partial cache can hold at least twelve complete layers, layers 0, 1, 19, and 2 are made fully resident within the same budget. Route replay found that these four layers save more routing and synchronization overhead than the global capacity they consume. Five resident layers performed worse because the fifth displaced too many experts from the shared pool.

For Metal shared buffers, positioned reads write directly into cache storage. Other supported buffers use staged uploads. Selected reads begin while the shared expert graph runs. During single-token decode, a mixed-hit layer can compute resident ranks while missing ranks are being read, then compute the missing ranks and combine both outputs in their original rank positions. `LLAMA_MOE_OVERLAP=0` and `LLAMA_MOE_SPLIT=0` remain private diagnostic environment variables.

The routed decode output fuses per-expert BF16 conversion, six-rank reduction, shared-expert addition, and final BF16 conversion into one Metal dispatch when its scratch layout is safe. Fully resident routed tensors use the normal graph and preserve original expert IDs. Streamed graphs sharing one model are serialized because they share mutable cache storage.

`LLAMA_MOE_TRACE=/path/to/routes.jsonl` records the cache layout, ordered routes, and read intervals for local diagnosis. Fully resident layers do not emit route events. Tracing adds I/O overhead and is not a supported cache configuration format.

## CED and images

Single-slot text serving defaults to CED prefill when the model supports it. The encoder processes the full prompt, and the decoder replays its final attention window. Every generated token still runs both halves. A routed cache therefore needs useful capacity for both encoder and decoder experts. Permanent encoder residency was removed because it reduced the capacity available to generation and did not make complete prompt processing resident.

Image input uses the V4.1 `deepseek41v` projector supplied with `--mmproj`. Image rows remain in reading order and have no V4 alignment padding. Image embeddings use the vision routing bias and bypass Engram. Text n-grams after an image stop at the image boundary. CED image chunks set the streaming phase to prefill before execution.

DSpark support exists elsewhere in the branch, but it is not part of the streaming configuration. Measurements on this M1 Ultra did not justify the memory and complexity cost, so current presets and validation must run without a draft model.

## Measured behavior on the M1 Ultra

The global pool, phase protection, mixed-hit overlap, fused output reduction, and four resident layers were first isolated with a fully resident control quant. These are architecture-level paths and do not select behavior by quantization type. On the control workload, the global pool improved repeated long-chat generation from about 13.50 to 14.36 tokens/s. Mixed-hit overlap improved a cold run from 9.60 to 10.09 tokens/s and a changed-topic repeat from 12.58 to 13.13 tokens/s. Four resident layers reached 14.53-14.54 tokens/s. The fully resident control reached 17.49 tokens/s, which shows that cache handler and Metal command-boundary overhead remain even without SSD misses.

The last MXFP4 comparison used the final global adaptive design and 8192-token batches:

| Routed cache | Cold long generation | Warm long generation | Warm short generation | Peak process memory |
| --- | ---: | ---: | ---: | ---: |
| 85 GiB | 5.44 tokens/s | 7.11 tokens/s | 11.92 tokens/s | 105.07 GiB |
| 88 GiB | 5.55 tokens/s | 8.46 tokens/s | 12.26 tokens/s | about 107.4 GiB |
| 95 GiB | 5.75 tokens/s | 13.06 tokens/s | 13.50 tokens/s | 111.59 GiB |

The 95 GiB warm result came from replaying the exact workload after its experts had entered the cache. It does not represent a cold or changed-topic chat. MXFP4 is too large for a useful cache hit rate at the desired memory margin, so the next target is a Q3_K routed quant with an 85 GiB cache. Its smaller bundles provide more resident expert slots at the same byte budget.

Four and eight read workers performed similarly after warmup; four remains fixed. A profile showed meaningful time in both SSD wait and Metal synchronization. Cache victim selection itself was negligible. A prototype that fused routed gate, up, and SwiGLU work either corrupted reused scratch or lost performance after adding the required guard, so it was removed.

The persistent V4.1 context allocation measured about 699 MiB at 128K context with one slot, including rollback and the encoder frontier. Context checkpoints are separate from `--cache-ram`; each checkpoint in that configuration can use about 590 MiB. Use physical footprint and swap to size the routed cache because startup RSS does not include all experts or lazy Engram rows that a conversation will touch.

## Calibration and quantization

For streamed calibration, `llama-imatrix` sees canonical source tensor dimensions even though the execution tensors contain only cache slots. V4.1 calibration uses full causal evaluation so that both halves receive importance data for every token.

```bash
./build/bin/llama-imatrix -m model.gguf -f calibration.txt \
    --stream-moe --moe-cache 81920 -ngl 99 -fa on --fit off \
    -c 8192 -b 8192 -ub 8192 --process-output --output-frequency 1 \
    -o imatrix.gguf
```

Calibration batches need more working memory than single-token generation. Only complete corpus chunks are evaluated, and imatrix checkpoints are replaced atomically. A zero-count expert has no measured importance; the quantizer substitutes unit importance for that expert.

The quantizer can requantize V4.1 Engram tables when a tensor override names them. Runtime lazy lookup then dequantizes only requested rows. Engram scale tensors remain in the GGUF for layout compatibility. Routed projections and the Engram KV projection are independently eligible for quantization.

Perplexity and reference-logit generation use full causal inference. Use `-c 512 -b 512 -ub 512` for the established V4.1 comparison so the tool evaluates one sequence. Supply the MXFP4 reference made with `--save-all-logits` through `--kl-divergence-base` when measuring a quantized model. Save the full command, build revision, memory samples, perplexity, and KLD output together.

## Limits

Row or tensor splitting, MTP loading, Windows streaming, and model loading without GGUF source files are rejected. Automatic CPU repacking is disabled for streamed experts. The global adaptive pool requires equal expert bundle sizes in every layer. LoRA tensors targeting nonresident routed weights are rejected because their canonical source descriptors do not expose resident buffers.

The low-level API exposes `llama_set_moe_phase`, `llama_get_moe_cache_stats`, and `llama_model_tensor_source`. Callers must set the phase before prefill or decode. The server, benchmark, and imatrix tool do this automatically.
