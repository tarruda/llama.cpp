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

After building GGML, compare the native packed activation rows and graph operation with the Python definitions:

```sh
.venv/bin/python examples/model-conversion/deepseek-v4.1/check-native-quant.py \
    --library build-dsv41/bin/libggml-base.dylib --backend cpu
.venv/bin/python examples/model-conversion/deepseek-v4.1/check-native-quant.py \
    --library build-dsv41/bin/libggml-base.dylib --backend metal
build-dsv41/bin/test-backend-ops test -o DSV41_ACT_QUANT -b MTL0
```

Adjust the library path and backend device name to the build. Omitting `--backend` checks only packed CPU rows. The script checks exact packed bytes and decoded values for MXFP4, NVFP4, and MXFP8, plus BF16 graph rounding. Its inputs include FP4/FP8 midpoint neighbors, E4M3 scale ties, signed zero, subnormal values, and random BF16 values across 40 exponents. The backend tests add contiguous and strided tensor layouts. These activation quantizers are separate from the existing weight quantizers; `DSV41_ACT_QUANT` returns F32 values and does not allocate packed cache storage.
