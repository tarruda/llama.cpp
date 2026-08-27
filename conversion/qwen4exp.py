from __future__ import annotations

import tempfile
from collections.abc import Callable, Iterable
from math import prod
from pathlib import Path
from typing import Any, BinaryIO, cast

import numpy as np
import torch
from torch import Tensor

from .base import LazyTorchTensor, ModelBase, TextModel, gguf, logger
from .qwen import _LinearAttentionVReorderBase, _Qwen35MRopeMixin


class _BufferedFileTensor:
    def __init__(self, path: Path, dtype: np.dtype, shape: tuple[int, ...]):
        self.path = path
        self.dtype = np.dtype(dtype)
        self.shape = shape
        self.nbytes = prod(shape) * self.dtype.itemsize
        if self.path.stat().st_size != self.nbytes:
            raise ValueError(f"Unexpected buffered tensor size in {self.path}")

    def tofile(self, output: BinaryIO) -> None:
        buffer = bytearray(16 * 1024 * 1024)
        view = memoryview(buffer)
        with open(self.path, "rb", buffering=0) as source:
            while n_read := source.readinto(buffer):
                offset = 0
                while offset < n_read:
                    n_written = output.write(view[offset:n_read])
                    if not n_written:
                        raise OSError(f"Could not write buffered tensor from {self.path}")
                    offset += n_written

    def byteswap(self, inplace: bool = False) -> _BufferedFileTensor:
        del inplace
        raise ValueError("Buffered PLE conversion does not support changing endianness")


@ModelBase.register("Qwen4ExpForConditionalGeneration", "Qwen4ExpForCausalLM")
@ModelBase.example("Qwen/Qwen3.8-Flash-Next")
class Qwen4ExpTextModel(_Qwen35MRopeMixin, _LinearAttentionVReorderBase):
    model_arch = gguf.MODEL_ARCH.QWEN4EXP
    supports_ngram_export = True

    _ple_marker = ".ple_embedding.ngram_embedding.shard_"
    _ple_constant_suffixes = (
        "ple_embedding.layer_multipliers",
        "ple_embedding.ngram_heads_offsets",
        "ple_embedding.ngram_heads_vocab_sizes",
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ple_constants: dict[str, list[int]] = {}
        self._ple_table: _BufferedFileTensor | None = None
        self._ple_temp_path: Path | None = None

        if self.hparams.get("ple_layer_ids") and not self.mtp_only:
            for suffix in self._ple_constant_suffixes:
                matches = [
                    (name, gen)
                    for name, gen in self.model_tensors.items()
                    if name.endswith(suffix)
                ]
                if len(matches) != 1:
                    raise ValueError(
                        f"Expected one PLE constant {suffix!r}, found {len(matches)}"
                    )
                name, gen = matches[0]
                data = LazyTorchTensor.to_eager(gen())
                if data.dtype != torch.int64:
                    raise ValueError(
                        f"PLE constant {name!r} has type {data.dtype}, expected torch.int64"
                    )
                self._ple_constants[suffix] = [int(value) for value in data.tolist()]
                del self.model_tensors[name]

    @classmethod
    def filter_tensors(
        cls, item: tuple[str, Callable[[], Tensor]]
    ) -> tuple[str, Callable[[], Tensor]] | None:
        item = TextModel.filter_tensors(item)
        if item is None:
            return None

        name, gen = item
        is_ple_shard = cls._ple_marker in name
        is_ple_constant = name.endswith(cls._ple_constant_suffixes)
        if cls.ngram_only and not (is_ple_shard or is_ple_constant):
            return None
        if cls.no_ngram and is_ple_shard:
            return None

        if name.startswith("model.mtp."):
            name = name.removeprefix("model.")

        if name.startswith("mtp."):
            if cls.no_mtp:
                return None
            if cls._original_block_count is None:
                raise ValueError(
                    "MTP tensors were indexed before the base layer count was set"
                )

            parts = name.split(".", 3)
            if len(parts) == 4 and parts[1] == "layers" and parts[2].isdecimal():
                mtp_idx = int(parts[2])
                name = f"model.layers.{cls._original_block_count + mtp_idx}.{parts[3]}"
                cls.opt_num_mtp_layers = max(cls.opt_num_mtp_layers, mtp_idx + 1)
            else:
                remapper = {
                    "mtp.fc_embedding": "nextn.fc_embedding",
                    "mtp.fc_hidden": "nextn.fc_hidden",
                    "mtp.pre_fc_norm_embedding": "enorm",
                    "mtp.pre_fc_norm_hidden": "hnorm",
                    "mtp.hyper_connection_mixer": "nextn.hyper_connection_mixer",
                }
                prefix = next(
                    (
                        prefix
                        for prefix in remapper
                        if name == prefix or name.startswith(prefix + ".")
                    ),
                    None,
                )
                if prefix is None:
                    raise ValueError(f"Unexpected Qwen4-Exp MTP tensor {name!r}")
                name = f"model.layers.{cls._original_block_count}.{remapper[prefix]}{name[len(prefix) :]}"
        elif cls.mtp_only:
            if name not in (
                "model.embed_tokens.weight",
                "embed_tokens.weight",
                "lm_head.weight",
            ):
                return None

        return name, gen

    def set_gguf_parameters(self):
        super().set_gguf_parameters()

        hc_count = int(self.hparams["hc_count"])
        hc_low_rank = int(self.hparams["hc_lowrank"])
        if hc_count <= 1 or hc_low_rank <= 0:
            raise ValueError(
                f"Invalid hyper-connection dimensions: count={hc_count}, low_rank={hc_low_rank}"
            )
        self.gguf_writer.add_hyper_connection_count(hc_count)
        self.gguf_writer.add_hyper_connection_low_rank(hc_low_rank)

        if self.hparams.get("output_gate_type") != "sigmoid":
            raise ValueError(
                "Qwen4-Exp requires a sigmoid linear-attention output gate"
            )
        if self.hparams.get("norm_topk_prob", True) is not True:
            raise ValueError("Qwen4-Exp requires normalized top-k expert weights")
        self.gguf_writer.add_expert_weights_norm(True)

        indexer_heads = int(self.hparams["indexer_n_heads"])
        indexer_kv_heads = int(self.hparams["indexer_kv_heads"])
        indexer_head_dim = int(self.hparams["indexer_head_dim"])
        indexer_budget = int(self.hparams["indexer_budget"])
        indexer_ratio = int(self.hparams["indexer_compress_ratio"])
        if indexer_heads <= 0 or indexer_kv_heads != 1 or indexer_head_dim <= 0:
            raise ValueError(
                "Qwen4-Exp requires positive QSA dimensions and one index key head"
            )
        if (
            indexer_budget <= 0
            or indexer_ratio <= 0
            or indexer_budget % indexer_ratio != 0
        ):
            raise ValueError(
                f"Invalid QSA budget or compression ratio: {indexer_budget}, {indexer_ratio}"
            )

        self.gguf_writer.add_indexer_head_count(indexer_heads)
        self.gguf_writer.add_indexer_key_length(indexer_head_dim)
        self.gguf_writer.add_indexer_top_k(indexer_budget)

        layer_types = list(self.hparams["layer_types"])
        if not self.no_mtp:
            n_mtp = self.block_count - int(self.hparams["num_hidden_layers"])
            mtp_types = list(self.hparams.get("mtp", {}).get("layer_types", []))
            if not mtp_types:
                mtp_types = ["full_attention"] * n_mtp
            layer_types.extend(mtp_types)
        if len(layer_types) != self.block_count:
            raise ValueError(
                f"Expected {self.block_count} layer types, found {len(layer_types)}"
            )
        full_attention_types = {"full_attention", "qwen_sparse_attention"}
        unknown_layer_types = (
            set(layer_types) - full_attention_types - {"linear_attention"}
        )
        if unknown_layer_types:
            raise ValueError(
                f"Unsupported Qwen4-Exp layer types: {sorted(unknown_layer_types)}"
            )
        self.gguf_writer.add_attention_recurrent_layers(
            [layer_type == "linear_attention" for layer_type in layer_types]
        )
        self.gguf_writer.add_attention_compress_ratios(
            [
                indexer_ratio if layer_type in full_attention_types else 0
                for layer_type in layer_types
            ]
        )

        if self.mtp_only:
            return

        ple_layers = [
            int(layer_id) - 1 for layer_id in self.hparams.get("ple_layer_ids", [])
        ]
        if not ple_layers:
            return
        if len(ple_layers) != 1:
            raise ValueError(
                f"Qwen4-Exp conversion currently requires one PLE layer, found {ple_layers}"
            )
        if ple_layers[0] < 0 or ple_layers[0] >= int(self.hparams["num_hidden_layers"]):
            raise ValueError(f"PLE layer is out of range: {ple_layers[0]}")

        ngram_size = int(self.hparams["ngram_size"])
        heads_per_ngram = int(self.hparams["heads_per_ngram"])
        ngram_heads = (ngram_size - 1) * heads_per_ngram
        ple_embed_dim = int(self.hparams["ple_embed_dim"])
        vocab_size_divisor = int(self.hparams["make_ngram_vocab_size_divisible_by"])
        if ngram_heads <= 0 or ple_embed_dim <= 0 or ple_embed_dim % ngram_heads != 0:
            raise ValueError(
                f"Invalid PLE embedding dimensions: {ple_embed_dim}, {ngram_heads}"
            )
        if vocab_size_divisor <= 0:
            raise ValueError(
                f"PLE vocabulary divisor must be positive, found {vocab_size_divisor}"
            )

        layer_multipliers = self._ple_constants["ple_embedding.layer_multipliers"]
        head_offsets = self._ple_constants["ple_embedding.ngram_heads_offsets"]
        head_vocab_sizes = self._ple_constants["ple_embedding.ngram_heads_vocab_sizes"]
        if len(layer_multipliers) != ngram_size:
            raise ValueError(
                f"Expected {ngram_size} PLE layer multipliers, found {len(layer_multipliers)}"
            )
        if len(head_offsets) != ngram_heads or len(head_vocab_sizes) != ngram_heads:
            raise ValueError(
                f"Expected {ngram_heads} PLE head offsets and vocabulary sizes, found "
                f"{len(head_offsets)} and {len(head_vocab_sizes)}"
            )
        if any(size <= 0 for size in head_vocab_sizes):
            raise ValueError("PLE head vocabulary sizes must be positive")
        expected_offsets = np.cumsum(
            [0, *head_vocab_sizes[:-1]], dtype=np.int64
        ).tolist()
        if head_offsets != expected_offsets:
            raise ValueError(f"PLE head offsets are not contiguous: {head_offsets}")

        eos_token_id = self.hparams.get("eos_token_id")
        if isinstance(eos_token_id, list):
            if len(eos_token_id) != 1:
                raise ValueError(
                    f"Qwen4-Exp PLE requires one EOS token, found {eos_token_id}"
                )
            eos_token_id = eos_token_id[0]
        if eos_token_id is None:
            raise ValueError("Qwen4-Exp PLE requires an EOS token")

        self.gguf_writer.add_per_layer_embedding_layers(ple_layers)
        self.gguf_writer.add_per_layer_embedding_ngram_size(ngram_size)
        self.gguf_writer.add_per_layer_embedding_heads_per_ngram(heads_per_ngram)
        self.gguf_writer.add_per_layer_embedding_vocab_size_divisor(vocab_size_divisor)
        self.gguf_writer.add_per_layer_embedding_conv_kernel(
            int(self.hparams["ple_conv_kernel_size"])
        )
        self.gguf_writer.add_per_layer_embedding_eos_token_id(int(eos_token_id))
        self.gguf_writer.add_embedding_length_per_layer_input(
            ple_embed_dim // ngram_heads
        )
        self.gguf_writer.add_per_layer_embedding_layer_multipliers(layer_multipliers)
        self.gguf_writer.add_per_layer_embedding_head_offsets(head_offsets)
        self.gguf_writer.add_per_layer_embedding_head_vocab_sizes(head_vocab_sizes)

    def _ple_qtype(self) -> gguf.GGMLQuantizationType:
        if self.ngram_only:
            if self.ngram_qtype is None:
                raise ValueError("N-gram-only conversion requires an explicit n-gram type")
            return self.ngram_qtype
        if self.ftype == gguf.LlamaFileType.ALL_F32:
            return gguf.GGMLQuantizationType.F32
        if self.ftype == gguf.LlamaFileType.MOSTLY_BF16:
            return gguf.GGMLQuantizationType.BF16
        if self.ftype == gguf.LlamaFileType.MOSTLY_Q8_0:
            return gguf.GGMLQuantizationType.Q8_0
        return gguf.GGMLQuantizationType.F16

    def _prepare_ple_table(self) -> None:
        if self.mtp_only or self.no_ngram:
            return

        shard_items: list[tuple[int, str, Callable[[], Tensor]]] = []
        for name, gen in self.model_tensors.items():
            if self._ple_marker not in name:
                continue
            index_text = name.split(self._ple_marker, 1)[1].split(".", 1)[0]
            if not index_text.isdecimal():
                raise ValueError(f"Invalid PLE shard name {name!r}")
            shard_items.append((int(index_text), name, gen))

        if not self.hparams.get("ple_layer_ids"):
            if shard_items:
                raise ValueError("Found PLE table shards without configured PLE layers")
            return

        shard_items.sort(key=lambda item: item[0])
        expected_parts = int(self.hparams["split_ngram_parts"])
        if [item[0] for item in shard_items] != list(range(expected_parts)):
            raise ValueError(
                f"Expected PLE shards 0 through {expected_parts - 1}, found {[item[0] for item in shard_items]}"
            )

        row_dim: int | None = None
        total_rows = 0
        for _, _, gen in shard_items:
            tensor = gen()
            if tensor.ndim != 2:
                raise ValueError("PLE table shards must be two-dimensional")
            if row_dim is None:
                row_dim = int(tensor.shape[1])
            elif int(tensor.shape[1]) != row_dim:
                raise ValueError("PLE table shards must have a common row width")
            total_rows += int(tensor.shape[0])
            del tensor

        if row_dim is None:
            raise ValueError("PLE table has no shards")
        ngram_heads = (int(self.hparams["ngram_size"]) - 1) * int(
            self.hparams["heads_per_ngram"]
        )
        expected_row_dim = int(self.hparams["ple_embed_dim"]) // ngram_heads
        if row_dim != expected_row_dim:
            raise ValueError(
                f"Expected PLE row width {expected_row_dim}, found {row_dim}"
            )

        head_vocab_sizes = self._ple_constants["ple_embedding.ngram_heads_vocab_sizes"]
        unpadded_rows = sum(head_vocab_sizes)
        divisor = int(self.hparams["make_ngram_vocab_size_divisible_by"])
        if divisor <= 0:
            raise ValueError(
                f"PLE vocabulary divisor must be positive, found {divisor}"
            )
        expected_rows = (unpadded_rows + divisor - 1) // divisor * divisor
        if total_rows != expected_rows:
            raise ValueError(
                f"Expected {expected_rows} PLE table rows, found {total_rows}"
            )

        qtype = self._ple_qtype()
        tmp_dir = self.fname_out if self.fname_out.is_dir() else self.fname_out.parent
        with tempfile.NamedTemporaryFile(
            prefix=".qwen4exp-ple-", suffix=".tmp", dir=tmp_dir, delete=False
        ) as fp:
            self._ple_temp_path = Path(fp.name)

        offset = 0
        encoded_dtype: np.dtype | None = None
        encoded_width: int | None = None
        with open(self._ple_temp_path, "wb", buffering=0) as table_file:
            for part, (_, _, gen) in enumerate(shard_items, start=1):
                logger.info(f"Assembling PLE shard {part}/{len(shard_items)} with buffered I/O")
                tensor = gen()
                rows = int(tensor.shape[0])
                source = LazyTorchTensor.to_eager(tensor).to(torch.float32).numpy()
                encoded = gguf.quants.quantize(source, qtype)
                if encoded.ndim != 2 or int(encoded.shape[0]) != rows:
                    raise ValueError("PLE shard produced an invalid encoded shape")
                if encoded_dtype is None:
                    encoded_dtype = encoded.dtype
                    encoded_width = int(encoded.shape[1])
                elif encoded.dtype != encoded_dtype or int(encoded.shape[1]) != encoded_width:
                    raise ValueError("PLE shards produced inconsistent encoded row widths")
                encoded.tofile(table_file)
                offset += rows
                del encoded, source, tensor

        if encoded_dtype is None or encoded_width is None or offset != total_rows:
            raise ValueError("PLE table assembly did not consume every row")
        table = _BufferedFileTensor(self._ple_temp_path, encoded_dtype, (total_rows, encoded_width))
        self._ple_table = table
        table_name = self.format_tensor_name(gguf.MODEL_TENSOR.PER_LAYER_TOKEN_EMBD)
        self.gguf_writer.add_tensor(table_name, cast(Any, table), raw_dtype=qtype)
        for _, name, _ in shard_items:
            del self.model_tensors[name]
        logger.info(
            f"Assembled {len(shard_items)} PLE shards into {table_name} with {total_rows} rows"
        )

    def modify_tensors(
        self, data_torch: Tensor, name: str, bid: int | None
    ) -> Iterable[tuple[str, Tensor]]:
        if name.endswith(".indexer.index_qk_proj.weight"):
            if bid is None:
                raise ValueError(f"QSA projection has no layer id: {name!r}")
            n_q = int(self.hparams["indexer_n_heads"]) * int(
                self.hparams["indexer_head_dim"]
            )
            n_k = int(self.hparams["indexer_kv_heads"]) * int(
                self.hparams["indexer_head_dim"]
            )
            if data_torch.ndim != 2 or int(data_torch.shape[0]) != n_q + n_k:
                raise ValueError(
                    f"Unexpected QSA projection shape for {name!r}: {tuple(data_torch.shape)}"
                )
            yield (
                self.format_tensor_name(gguf.MODEL_TENSOR.INDEXER_Q_PROJ, bid),
                data_torch[:n_q],
            )
            yield (
                self.format_tensor_name(gguf.MODEL_TENSOR.INDEXER_K_PROJ, bid),
                data_torch[n_q:],
            )
            return

        if name.endswith(
            (".ple.norm_key.weight", ".ple.norm_query.weight", ".ple.norm_conv.weight")
        ):
            data_torch = data_torch + 1

        yield from super().modify_tensors(data_torch, name, bid)

    def prepare_tensors(self):
        self._prepare_ple_table()
        super().prepare_tensors()

    def prepare_metadata(self, vocab_only: bool):
        from_dir = self.fname_out.is_dir()
        super().prepare_metadata(vocab_only=vocab_only)

        if self.ngram_only and from_dir:
            self.fname_out = self.fname_out.with_name(f"ngram-{self.fname_out.name}")

    def write(self):
        try:
            super().write()
        finally:
            self._ple_table = None
            if self._ple_temp_path is not None:
                self._ple_temp_path.unlink(missing_ok=True)
