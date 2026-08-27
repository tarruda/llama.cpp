# Qwen3.8 Flash Next resident vs disk-backed PLE benchmarks

## Goal

Measure the cost of keeping the separate Qwen3.8 Flash Next per-layer token embedding (PLE, or n-gram) table on the local SSD instead of loading it into unified memory. Also measure the effect of the separate MTP draft model and verify that all configurations produce the same output.

These tests target local, single-user inference on a 128 GiB Apple M1 Ultra. The trunk is loaded eagerly without mmap. The PLE sidecar is either:

- `resident`: read the complete sidecar into memory during model load.
- `read`: keep the sidecar on the local SSD and synchronously read selected rows on demand.

The runtime `read` path does not mmap the PLE. It seeks to each selected row, reads its encoded bytes, dequantizes the row to F32, and uploads the staging buffer used by the graph.

## Build and artifacts

Tests were run on 2026-08-27 from branch `mmap-bf16-imatrix`, commit `675b12931` (build 10670), with additional uncommitted worktree changes described below.

| Artifact | Path | Size | Contents |
| --- | --- | ---: | --- |
| Trunk | `/Users/thiago/qwen-3.8-next/Qwen3.8-Flash-Next-IQ4_NL.gguf` | 68.384 GiB | 1,223 tensors; PLE excluded |
| PLE | `/Users/thiago/qwen-3.8-next/Qwen3.8-Flash-Next-ngram-Q5_1.gguf` | 35.773 GiB | One Q5_1 tensor, `per_layer_token_embd.weight`, shape `[160, 320001536]` |
| MTP | `/Users/thiago/qwen-3.8-next/Qwen3.8-Flash-Next-MTP-Q8_0.gguf` | 3.852 GiB | 35 tensors: 24 Q8_0 and 11 F32 |
| Vision projector | `/Users/thiago/qwen-3.8-next/Qwen3.8-Flash-Next-mmproj-Q8_0.gguf` | 0.574 GiB | 334 tensors |

The trunk tensor types are 144 IQ4_NL, 666 Q8_0, 389 F32, and 24 BF16. Its GGUF metadata was checked to confirm that `per_layer_token_embd.weight` is absent.

A Q8_0 PLE sidecar was also produced and validated at 50.674 GiB, but these benchmarks use Q5_1.

## Run script

The personal server script is `/Users/thiago/run-qwen-3.8-flash-next.sh`. Its relevant environment controls are:

- `NGRAM_LOAD_MODE=resident|read`, default `resident`.
- `USE_MTP=0|1`, default `0`.
- `LOAD_MODE=<mode>`, default `none` for eager, non-mmap trunk loading.
- `CTX_SIZE=<tokens>`, default 262144; tests used 4096.

When `USE_MTP=1`, the script adds:

```text
--spec-type draft-mtp
--model-draft /Users/thiago/qwen-3.8-next/Qwen3.8-Flash-Next-MTP-Q8_0.gguf
--spec-draft-n-max 3
--spec-draft-p-min 0.0
```

Example launches:

```sh
CTX_SIZE=4096 HOST=127.0.0.1 PORT=18080 USE_MTP=0 NGRAM_LOAD_MODE=resident /Users/thiago/run-qwen-3.8-flash-next.sh
CTX_SIZE=4096 HOST=127.0.0.1 PORT=18080 USE_MTP=0 NGRAM_LOAD_MODE=read /Users/thiago/run-qwen-3.8-flash-next.sh
CTX_SIZE=4096 HOST=127.0.0.1 PORT=18080 USE_MTP=1 NGRAM_LOAD_MODE=resident /Users/thiago/run-qwen-3.8-flash-next.sh
CTX_SIZE=4096 HOST=127.0.0.1 PORT=18080 USE_MTP=1 NGRAM_LOAD_MODE=read /Users/thiago/run-qwen-3.8-flash-next.sh
```

## llama-server methodology

Each of the four configurations was freshly loaded. `--no-warmup` was enabled, so each configuration first received this smoke request:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Reply with exactly SMOKE_OK and nothing else."
    }
  ],
  "max_tokens": 32,
  "temperature": 0,
  "seed": 42,
  "stream": false,
  "chat_template_kwargs": {
    "enable_thinking": false
  }
}
```

The longer deterministic request was:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Output the integers from 1 upward in ascending order, separated by single spaces. Output numbers only and continue until stopped."
    }
  ],
  "max_tokens": 256,
  "temperature": 0,
  "seed": 42,
  "stream": false,
  "chat_template_kwargs": {
    "enable_thinking": false
  }
}
```

The benchmark prompt used 36 tokens and generation stopped at the 256-token limit.

## llama-server results

| PLE mode | MTP | Load time | Prompt tok/s | Generation tok/s | Draft accepted | Observed RSS |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| resident | no | 30.44 s | 100.99 | 38.96 | n/a | 86.7 GiB |
| read | no | 20.36 s | 86.14 | 37.03 | n/a | 70.4 GiB |
| resident | yes | 32.38 s | 91.79 | 48.87 | 191/191 | 86.3 GiB |
| read | yes | 21.58 s | 83.17 | 46.35 | 191/191 | 75.6 GiB |

The prompt rates come from a short 36-token prompt and are noisier than the generation results. The `llama-bench` PP512 results below are the better prompt-processing comparison.

Key deltas:

- Disk-read PLE reduced non-MTP generation throughput by 5.0% in this test.
- Disk-read PLE reduced MTP generation throughput by 5.2%.
- MTP increased resident generation throughput by 25.4%.
- MTP increased disk-read generation throughput by 25.2%.
- Resident startup took about 10 to 11 seconds longer because it read the full 35.773 GiB PLE sidecar.

The observed process RSS is not a reliable measure of the logical allocation difference on macOS because unified-memory accounting and memory compression changed between runs. Resident mode logically owns the complete 35.773 GiB PLE buffer even though the RSS delta was smaller.

## Output sanity

All four smoke requests returned HTTP 200 with exactly:

```text
SMOKE_OK
```

All four long requests produced byte-identical output with SHA-256:

```text
89002d20d1668d09bdca8f9104a4762a2bccdd64b9e3325c2b01a6a2e9529a9e
```

The output was the correct ascending integer sequence through 88, followed by the truncated prefix `8` when the 256-token limit was reached. MTP accepted 191/191 draft tokens for the long sequence and 3/3 for the smoke request. Greedy MTP output therefore matched non-MTP output exactly in this test.

The 100% MTP acceptance rate is specific to this highly predictable numeric sequence. It should not be treated as representative of natural-language workloads.

## MTP sidecar loader fix

The first MTP smoke start failed with:

```text
Qwen4 model has no n-gram table metadata
```

The common draft-parameter helper copied the target model's `path_ngram` into the separate MTP-only model parameters. Qwen4 correctly rejects an n-gram sidecar on an MTP-only GGUF.

The fix in `common/speculative.cpp` clears `result.path_ngram` when constructing parameters for a separate draft model:

```cpp
result.model = params_spec.mparams;
result.path_ngram.clear();
```

After rebuilding `llama-server`, both resident and disk-read MTP configurations loaded and completed the curl checks.

## llama-bench methodology

The exact command was:

```sh
build/bin/llama-bench \
  -m /Users/thiago/qwen-3.8-next/Qwen3.8-Flash-Next-IQ4_NL.gguf \
  --model-ngram /Users/thiago/qwen-3.8-next/Qwen3.8-Flash-Next-ngram-Q5_1.gguf \
  -lm none \
  --ngram-load-mode resident,read \
  -n 128 \
  -p 512 \
  -d 0,10000,20000 \
  --progress \
  -o jsonl
```

Other settings were llama-bench defaults:

- Five measured repetitions per row, with a warmup.
- Batch size 2048 and ubatch size 512.
- 16 threads.
- All layers eligible for Metal offload (`-ngl -1`).
- F16 K and V caches.
- Eager trunk loading (`-lm none`).

`-d` is context depth. For nonzero depths, llama-bench prepares that depth before measuring the PP512 or TG128 operation and reuses the prepared state across the five samples.

## llama-bench summary

| Depth | Resident PP512 | Read PP512 | Read penalty | Resident TG128 | Read TG128 | Read penalty |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 534.99 | 277.49 | -48.1% | 39.34 | 36.35 | -7.6% |
| 10000 | 447.41 | 251.85 | -43.7% | 34.41 | 32.07 | -6.8% |
| 20000 | 425.95 | 244.83 | -42.5% | 33.93 | 31.73 | -6.5% |

Disk reads have a much larger effect on batched prompt processing than on single-token generation. For this local SSD and workload, disk-backed TG128 remained within 6.5% to 7.6% of resident speed through a 20k-token depth. PP512 was 42.5% to 48.1% slower.

## llama-bench raw samples

Values are tokens per second. Each row is `mean +/- sample standard deviation`, followed by the five raw samples.

| PLE mode | Depth | Test | Mean +/- SD | Samples |
| --- | ---: | --- | ---: | --- |
| resident | 0 | PP512 | 534.990659 +/- 9.330860 | 540.487, 542.328, 540.523, 531.507, 520.109 |
| resident | 0 | TG128 | 39.341042 +/- 0.130432 | 39.4007, 39.4953, 39.3249, 39.1400, 39.3444 |
| resident | 10000 | PP512 | 447.411854 +/- 1.672755 | 449.127, 446.347, 445.125, 448.752, 447.709 |
| resident | 10000 | TG128 | 34.410634 +/- 0.050419 | 34.3546, 34.4413, 34.4817, 34.3894, 34.3862 |
| resident | 20000 | PP512 | 425.948363 +/- 3.383618 | 431.603, 423.014, 426.396, 424.380, 424.349 |
| resident | 20000 | TG128 | 33.932075 +/- 0.234034 | 34.1926, 33.9789, 34.1086, 33.7101, 33.6702 |
| read | 0 | PP512 | 277.491488 +/- 0.412401 | 277.101, 277.535, 278.176, 277.314, 277.332 |
| read | 0 | TG128 | 36.351050 +/- 0.093819 | 36.2678, 36.2491, 36.3498, 36.4599, 36.4287 |
| read | 10000 | PP512 | 251.848065 +/- 1.695024 | 254.376, 252.045, 252.219, 250.042, 250.558 |
| read | 10000 | TG128 | 32.067078 +/- 0.158360 | 32.1585, 32.1136, 32.1600, 31.7866, 32.1167 |
| read | 20000 | PP512 | 244.828704 +/- 2.065159 | 247.596, 242.158, 246.024, 244.241, 244.125 |
| read | 20000 | TG128 | 31.725878 +/- 0.014747 | 31.7438, 31.7226, 31.7384, 31.7101, 31.7144 |

## Why llama-bench has no MTP rows

The current llama-bench supports `--model-ngram` and `--ngram-load-mode`, but it has no draft-model, speculative type, sampling, target-verification, or acceptance-accounting options. Its TG loop directly decodes the target model.

Loading the MTP GGUF as an ordinary llama-bench model would measure the wrong graph and would not represent end-to-end speculative throughput. An honest MTP implementation in llama-bench would require a larger benchmark-harness design that drives draft generation, target verification, sampling, and acceptance. The four-way MTP comparison is therefore covered by the llama-server curl matrix, not by invented llama-bench rows.

## Imatrix and the Q5_1 PLE

The Q5_1 PLE conversion did not use `imatrix.gguf`. The buffered converter quantized each 160-value row with gguf-py's ordinary Q5_1 encoder.

The downloaded imatrix contains no `per_layer_token_embd.weight` entry. Its only PLE-named entries are:

```text
blk.1.ple_key.weight.in_sum2
blk.1.ple_key.weight.counts
blk.1.ple_value.weight.in_sum2
blk.1.ple_value.weight.counts
```

Those entries describe the smaller `ple_key` and `ple_value` matrix weights, not the 320-million-row lookup table.

llama.cpp's native Q5_1 quantizer can use a per-column importance vector. A custom 160-element sensitivity vector could therefore influence Q5_1 scale/min selection in principle. Standard imatrix collection cannot generate that signal for `per_layer_token_embd.weight` because PLE uses indexed row lookup rather than a matrix multiplication.

The expected benefit also decreases at higher BPW. Q5_1 already uses 5-bit values plus scale and minimum for every 32-value block. Row-frequency counts alone would not change each row's optimum because rows are independently quantized at fixed precision. A custom sensitivity experiment is possible, but Q5_1 versus Q8_0 KLD/perplexity and inference comparisons should come first.

## Conclusions

1. The separate Q5_1 PLE works in both resident and true read-on-demand modes.
2. Disk-backed PLE is practical for single-user token generation on this local SSD. It cost about 5% in the server sequence test and 6.5% to 7.6% in llama-bench TG128.
3. Disk-backed PLE is substantially slower for large batched prompt ingestion. PP512 lost 42.5% to 48.1%.
4. Resident mode saves prompt and generation time but adds a 35.773 GiB logical allocation and about 10 seconds of startup I/O.
5. The Q8_0 MTP sidecar worked after preventing the target PLE path from leaking into draft-model parameters. It improved the predictable sequence benchmark by about 25% with 100% acceptance and preserved output exactly.
6. MTP gains need additional natural-language measurements because acceptance rate controls the benefit.
7. The personal run script defaults to resident PLE and no MTP. Use `NGRAM_LOAD_MODE=read` when memory capacity matters, and `USE_MTP=1` to enable the separate MTP model.
