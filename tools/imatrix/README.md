# llama.cpp/tools/imatrix

Compute an importance matrix for a model and given text dataset. Can be used during quantization to enhance the quality of the quantized models.
More information is available in <https://github.com/ggml-org/llama.cpp/pull/4861>.

## Usage

```
./llama-imatrix \
    -m model.gguf -f some-text.txt [-o imatrix.gguf] [--output-format {gguf,dat}] [--no-ppl] \
    [--process-output] [--chunk 123] [--save-frequency 0] [--output-frequency 10] \
    [--in-file imatrix-prev-0.gguf --in-file imatrix-prev-1.gguf ...] [--parse-special] \
    [--show-statistics] [...]
```

Here `-m | --model` with a model name and `-f | --file` with a file containing calibration data (such as e.g. `wiki.train.raw`) are mandatory.
The parameters in square brackets are optional and have the following meaning:

* `-h | --help` shows usage information and exits.
* `-lv | --verbosity` specifies the verbosity level. If set to `0`, no output other than the perplexity of the processed chunks will be generated. If set to `1`, each time the results are saved a message is written to `stderr`. If `>=2`, a message is output each time data is collected for any tensor. Default verbosity level is `1`.
* `-o | --output-file` specifies the name of the file where the computed data will be stored. If missing `imatrix.gguf` is used.
* `-ofreq | --output-frequency` specifies how often the so far computed result is saved to disk. Default is 10 (i.e., every 10 chunks)
* `--output-format` specifies the output format of the generated imatrix file. Either "gguf", or "dat" (the legacy format). Defaults to "gguf".
* `--save-frequency` specifies how often to save a copy of the imatrix in a separate file. Default is 0 (i.e., never)
* `--process-output` specifies if data will be collected for the `output.weight` tensor. Typically, it is better not to utilize the importance matrix when quantizing `output.weight`, so this is set to `false` by default.
* `--in-file` one or more existing imatrix files to load and combine. Useful for merging files from multiple runs/datasets.
* `--parse-special` enables parsing of special tokens (e.g., `<|im_start|>` in some models). Useful for models with custom tokenizers.
* `--chunk | --from-chunk` to skip the first `n` chunks of tokens from the input data. Useful for resuming or skipping initial low-quality data.
* `--chunks` maximum number of chunks to process. Default is -1 for all available chunks.
* `--no-ppl` disables the calculation of perplexity for the processed chunks. Useful if you want to speed up the processing and do not care about perplexity.
* `--show-statistics` displays imatrix file's statistics.

For faster computation, make sure to use GPU offloading via the `-ngl | --n-gpu-layers` argument.

### Routed expert profiles

`--moe-profile-output experts.json` records original expert IDs and selection counts per layer. These counts measure routing frequency, not the effect of removing an expert. They are separate from the activation importance values used by the quantizer.

Calibration uses full causal evaluation so that both halves of DeepSeek V4.1 receive importance data for every calibration token. To also measure inference traffic, add `--moe-profile-generate 128 --moe-profile-prompts 8 --moe-profile-prompt-tokens 1024`. After calibration, the tool prefills evenly spaced excerpts from the same corpus and samples continuations with the configured sampler. These are raw text continuations, not chat-template prompts. The JSON records separate `calibration`, `prefill`, and `decode` counts, token observations, expert coverage, and sampling settings. Generation profiling does not change the imatrix.

For DeepSeek V4.1, profiling uses CED prefill: encoder layers process the prompt, decoder layers replay its final window, and both halves process each generated token. Counts describe the work actually executed, including that replay. An early end-of-generation token can shorten a sample; inspect the recorded observations when assessing coverage. Use a representative corpus and enough sampled continuations before treating a frequency ranking as stable.

```bash
./build/bin/llama-imatrix -m model.gguf -f calibration.txt \
    --stream-moe --moe-cache 81920 -ngl 99 -fa on \
    -c 4096 -b 4096 -ub 4096 --process-output \
    -o imatrix.gguf --moe-profile-output experts.json \
    --moe-profile-generate 128 --moe-profile-prompts 8 --moe-profile-prompt-tokens 1024 --seed 41
```

Streaming forces `--load-mode none --lazy-mode on` and uses one calibration sequence. The cache budget is in MiB and excludes other weights, activations, output buffers, context state, and lazy row caches. Leave memory for these allocations, especially with large calibration batches. Only complete corpus chunks are calibrated; a trailing partial chunk is skipped. Checkpoints count completed corpus chunks and GGUF output is replaced atomically after a successful write.

`--in-file` merges imatrix values and counts only. Routing JSON from earlier runs is not merged: a new JSON describes observations from the current invocation. Profiles are compatible across quantizations with the same architecture and expert topology, but changing quantization can change routing frequencies. See [routed expert streaming](../../docs/development/routed-expert-streaming.md) for residency and pin settings.

Recent versions of `llama-imatrix` store data in GGUF format by default. For the legacy format, use an extension other than `.gguf` when saving the output file. More information is available in <https://github.com/ggml-org/llama.cpp/pull/9400>.

## Examples

```bash
# generate importance matrix using default filename (imatrix.gguf), offloading 99 layers to GPU
./llama-imatrix -m ggml-model-f16.gguf -f calibration-data.txt -ngl 99

# use the imatrix to perform a Q4_K_M quantization
./llama-quantize --imatrix imatrix.gguf ggml-model-f16.gguf ./ggml-model-q4_k_m.gguf q4_k_m
```

```bash
# generate and save the imatrix using legacy format
./llama-imatrix -m ggml-model-f16.gguf -f calibration-data.txt --output-format dat -o imatrix-legcy-format.dat -ngl 99
```

```bash
# convert legacy (binary) imatrix format to new (GGUF) format
./llama-imatrix --in-file imatrix-legacy-format.dat -o imatrix-new-format.gguf
```

```bash
# convert new (GGUF) imatrix format to legacy (binary) format
./llama-imatrix --in-file imatrix-new-format.gguf --output-format dat -o imatrix-legacy-format.dat
```

```bash
# combine existing imatrices
./llama-imatrix --in-file imatrix-prev-0.gguf --in-file imatrix-prev-1.gguf -o imatrix-combined.gguf
```

```bash
# skip first 5 chunks, save intermediates every 20 chunks and snapshots every 50, parsing special tokens
./llama-imatrix -m ggml-model-f16.gguf -f calibration-data.txt --chunk 5 --output-frequency 20 --save-frequency 50 --parse-special
```

```bash
# analyse imatrix file and display summary statistics instead of running inference
./llama-imatrix --in-file imatrix.gguf --show-statistics
```

`--show-statistics` will display the following statistics:

#### Per tensor

* Σ(Act²): sum of all squared activations (the importance scores)
* Min & Max: minimum and maximum squared activations values
* μ & σ: Squared activations' mean and standard deviation
* % Active: proportion of elements whose average squared activation exceeds a small threshold (1e-5). Helpful to determine how alive/dormant the tensor is during inference
* N: number of squared activations
* Entropy: entropy of the squared activation distribution, in bits (standard Shannon entropy measurement) $S = -\sum_{i=1}^N p_i \log_2 p_i$
* E (norm): Normalized entropy. $E(norm)=\frac{-\sum_{i=1}^N p_i \log_2 p_i}{log_2 N}$. These two metrics can be used to determine how well a prompt "exercises" the model's capabilities
* ZD Score: z-score distribution as described in _3.1 Layer Importance Scores_ of [Layer-Wise Quantization](https://arxiv.org/abs/2406.17415)
* CosSim: cosine similarity with respect to the previous layer's tensor. Useful to determine how similar the squared activations of the current layer are to the previous layer's squared activations.

#### Per layer

Weighted averages of Σ(Act²), ZD Score and CosSim are also calculated.

#### Important note on the computed Statistics

When using these statistics, please note that they are computed on the squared activations, **not on the actual (raw) activations**.
Whilst the results are still useful, they're less reliable than using the raw values, and in the case of the cosine similarity, could be misleading if the tensor contains opposite vectors.
