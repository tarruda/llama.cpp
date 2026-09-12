from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Callable, Iterable, TYPE_CHECKING

import numpy as np
import torch

from .base import LazyTorchTensor, MmprojModel, ModelBase, TextModel, gguf, logger
from .deepseek import DeepseekV4FlashVisionModel, DeepseekV4Model

if TYPE_CHECKING:
    from torch import Tensor


def build_engram_token_map(dir_model: Path) -> list[int]:
    from tokenizers import Regex, Tokenizer, normalizers

    tokenizer = Tokenizer.from_file(str(dir_model / "tokenizer.json"))
    normalizer = normalizers.Sequence([
        normalizers.NFKC(), normalizers.NFD(), normalizers.StripAccents(), normalizers.Lowercase(),
        normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
        normalizers.Replace(Regex(r"^ $"), "\ue000"), normalizers.Strip(), normalizers.Replace("\ue000", " "),
    ])
    lookup: list[int] = []
    ids: dict[str, int] = {}
    for token_id in range(tokenizer.get_vocab_size()):
        text = tokenizer.decode([token_id], skip_special_tokens=False)
        key = tokenizer.id_to_token(token_id) if "\ufffd" in text else normalizer.normalize_str(text) or text
        lookup.append(ids.setdefault(key, len(ids)))
    return lookup


@ModelBase.register("DeepseekV41ForCausalLM")
@ModelBase.example("deepseek-ai/DeepSeek-V4.1-Flash")
class DeepseekV41Model(DeepseekV4Model):
    model_arch = gguf.MODEL_ARCH.DEEPSEEK41
    supports_mtp_export = False

    def index_tensors(self, remote_hf_model_id: str | None = None) -> dict[str, Callable[[], Tensor]]:
        return ModelBase.index_tensors(self, remote_hf_model_id=remote_hf_model_id)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        if item[0].startswith(("mtp.", "aligner.", "image_")):
            return None
        return TextModel.filter_tensors(item)

    def fp8_weight_block_size(self) -> int:
        block = self.hparams.get("quantization_config", {}).get("weight_block_size", [32, 32])
        if block != [32, 32]:
            raise ValueError(f"DeepSeek-V4.1 expects FP8 weight blocks [32, 32], got {block}")
        return 32

    def dequant_model(self):
        block = self.fp8_weight_block_size()
        for name, gen in self.model_tensors.items():
            weight = gen()
            if weight.dtype not in self._float8_dtypes():
                continue
            scale = self.model_tensors.get(name.removesuffix(".weight") + ".scale")
            if not name.endswith(".weight") or scale is None or len(weight.shape) != 2:
                raise ValueError(f"Missing FP8 block scales for {name}")
            expected = tuple((dim + block - 1) // block for dim in weight.shape)
            if tuple(scale().shape) != expected:
                raise ValueError(f"Invalid FP8 scale shape for {name}: expected {expected}, got {tuple(scale().shape)}")
        super().dequant_model()

    def tensor_force_quant(self, name: str, new_name: str, bid: int | None, n_dims: int) -> gguf.GGMLQuantizationType | bool:
        if name in self._dsv4_fp8_dequantized and n_dims >= 2:
            return gguf.GGMLQuantizationType.Q8_0 if self._fp8_as_q8 else gguf.GGMLQuantizationType.BF16
        return super().tensor_force_quant(name, new_name, bid, n_dims)

    def set_vocab(self):
        super().set_vocab()
        template = Path(__file__).parent.parent / "models/templates/deepseek-ai-DeepSeek-V4.1.jinja"
        self.gguf_writer.add_chat_template(template.read_text(encoding="utf-8"))

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        h = self.hparams
        self.gguf_writer.add_embedding_length_out(h["hidden_size"])
        self.gguf_writer.add_attention_kv_source_layers(h["kv_source_layer_ids"])
        self.gguf_writer.add_attention_index_source_layers(h["index_source_layer_ids"])
        self.gguf_writer.add_indexer_candidate_source_layer(h["candidate_source_layer_id"])
        self.gguf_writer.add_indexer_candidate_topk_blocks(h["candidate_topk_blocks"])
        self.gguf_writer.add_indexer_candidate_block_size(h["candidate_block_size"])
        self._set_engram_parameters()

    def _set_engram_parameters(self):
        h = self.hparams
        layers = h.get("engram_layer_ids", [])
        if not layers:
            return
        counts = h["engram_num_embeddings"]
        if len(layers) != len(counts):
            raise ValueError("Engram layer and table counts differ")
        token_map = build_engram_token_map(self.dir_model)
        vocab_size = len(set(token_map))
        if len(token_map) != h["vocab_size"] or vocab_size != h["engram_compressed_vocab_size"]:
            raise ValueError("Engram tokenizer does not match the configured vocabulary")

        seen: set[int] = set()
        sizes, offsets, multipliers = [], [], []
        bound = max(1, (np.iinfo(np.int64).max // vocab_size) // 2)
        for layer_id, expected_rows in zip(layers, counts):
            offset = 0
            for _ in range(h["engram_max_ngram_size"] - 1):
                prime = h["engram_vocab_size"] - 1
                for _ in range(h["engram_n_heads"]):
                    prime += 1
                    while prime in seen or prime < 2 or any(prime % d == 0 for d in range(2, math.isqrt(prime) + 1)):
                        prime += 1
                    seen.add(prime)
                    sizes.append(prime)
                    offsets.append(offset)
                    offset += prime
            if offset != expected_rows:
                raise ValueError(f"Engram layer {layer_id}: bucket layout has {offset} rows, expected {expected_rows}")
            rng = np.random.default_rng(10007 * layer_id)
            multipliers.extend((2 * rng.integers(0, bound, size=h["engram_max_ngram_size"], dtype=np.int64) + 1).tolist())

        w = self.gguf_writer
        w.add_engram_layers(layers)
        w.add_engram_ngram_size(h["engram_max_ngram_size"])
        w.add_engram_head_count(h["engram_n_heads"])
        w.add_engram_head_dim(h["engram_head_dim"])
        w.add_engram_embedding_counts(counts)
        w.add_engram_head_offsets(offsets)
        w.add_engram_head_bucket_sizes(sizes)
        w.add_engram_multipliers(multipliers)
        w.add_engram_token_map(token_map)
        w.add_engram_compressed_vocab_size(vocab_size)
        # The runtime applies token_map to this ID, as it does to input tokens.
        w.add_engram_pad_token_id(h["engram_pad_token_id"])

    def _write_engram_tables(self):
        h = self.hparams
        for bid, rows in zip(h.get("engram_layer_ids", []), h.get("engram_num_embeddings", [])):
            for suffix, key, width, dtype in (
                ("weight", gguf.MODEL_TENSOR.ENGRAM_EMBD, h["engram_head_dim"], torch.float8_e4m3fn),
                ("scale", gguf.MODEL_TENSOR.ENGRAM_EMBD_SCALE, h["engram_head_dim"] // 32, torch.uint8),
            ):
                source = f"layers.{bid}.engram.embed.{suffix}"
                gen = self.model_tensors.pop(source)
                tensor = gen()
                valid_dtype = tensor.dtype == dtype or (suffix == "scale" and tensor.dtype == getattr(torch, "float8_e8m0fnu", None))
                if not valid_dtype or tuple(tensor.shape) != (rows, width):
                    raise ValueError(f"Invalid Engram tensor {source}: {tensor.dtype}, {tuple(tensor.shape)}")
                if self.remote_hf_model_id is not None:
                    raise ValueError("Convert Engram tables from a local safetensors checkpoint")
                table = LazyTorchTensor.to_eager(tensor).view(torch.uint8).numpy()
                chunks = [lambda start=start, t=table: t[start:start + (1 << 18)] for start in range(0, rows, 1 << 18)]
                data = gguf.LazyChunkedTensor(chunks, (rows, width), np.int8)
                name = self.format_tensor_name(key, bid)
                logger.info("%s: preserving Engram bytes as I8, shape = %s", name, (rows, width))
                self.gguf_writer.add_tensor(name, data)

    def _write_mxfp4_expert_tensor(self, bid: int, proj: str, tensor_key: gguf.MODEL_TENSOR) -> list[str]:
        consumed, chunks = [], []
        byte_shape = None
        for eid in range(self.hparams["n_routed_experts"]):
            prefix = f"layers.{bid}.ffn.experts.{eid}.{proj}"
            weight, scale = self.model_tensors[prefix + ".weight"], self.model_tensors[prefix + ".scale"]
            shape = tuple(weight().shape)
            if weight().dtype != torch.int8 or len(shape) != 2 or shape[1] % 16 or tuple(scale().shape) != (shape[0], shape[1] // 16):
                raise ValueError(f"Invalid packed MXFP4 tensor {prefix}")
            packed_shape = (shape[0], shape[1] // 16 * 17)
            if byte_shape is not None and byte_shape != packed_shape:
                raise ValueError(f"Expert shapes differ in layer {bid}")
            byte_shape = packed_shape
            chunks.append(lambda w=weight, s=scale: self.repack_mxfp4_blocks(LazyTorchTensor.to_eager(w()), LazyTorchTensor.to_eager(s())))
            consumed.extend((prefix + ".weight", prefix + ".scale"))
        assert byte_shape is not None
        data = gguf.LazyChunkedTensor(chunks, (len(chunks), *byte_shape), np.uint8)
        name = self.format_tensor_name(tensor_key, bid)
        logger.info("%s: preserving routed MXFP4 in %d expert chunks", name, len(chunks))
        self.gguf_writer.add_tensor(name, data, raw_dtype=gguf.GGMLQuantizationType.MXFP4)
        return consumed

    def generate_extra_tensors(self) -> Iterable[tuple[str, Tensor]]:
        if self._dsv4_mxfp4_generated:
            return ()
        for bid in range(self.block_count):
            for proj, key in (("w1", gguf.MODEL_TENSOR.FFN_GATE_EXP), ("w2", gguf.MODEL_TENSOR.FFN_DOWN_EXP), ("w3", gguf.MODEL_TENSOR.FFN_UP_EXP)):
                for name in self._write_mxfp4_expert_tensor(bid, proj, key):
                    del self.model_tensors[name]
        self._dsv4_mxfp4_generated = True
        return ()

    def prepare_tensors(self):
        self._write_engram_tables()
        super().prepare_tensors()

    def _map_dsv4_tensor_name(self, name: str, bid: int | None) -> tuple[gguf.MODEL_TENSOR, str]:
        extra = {
            "attn.kv_norm.weight": gguf.MODEL_TENSOR.ATTN_KV_A_NORM,
            "attn.indexer.wk.weight": gguf.MODEL_TENSOR.INDEXER_ATTN_K,
            "attn.indexer.k_norm.weight": gguf.MODEL_TENSOR.INDEXER_K_NORM,
            "engram.wkv.weight": gguf.MODEL_TENSOR.ENGRAM_KV,
            "engram.q_weight": gguf.MODEL_TENSOR.ENGRAM_Q_WEIGHT,
            "engram.k_weight": gguf.MODEL_TENSOR.ENGRAM_K_WEIGHT,
        }
        if bid is not None and (key := extra.get(name.removeprefix(f"layers.{bid}."))) is not None:
            return key, ".weight"
        return super()._map_dsv4_tensor_name(name, bid)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        key, suffix = self._map_dsv4_tensor_name(name, bid)
        yield self.format_tensor_name(key, bid, suffix), data_torch


@ModelBase.register("DeepseekV41DSparkModel")
class DeepseekV41DSparkModel(DeepseekV41Model):
    model_arch = gguf.MODEL_ARCH.DFLASH

    _root_map = {
        "main_proj.weight": gguf.MODEL_TENSOR.FC,
        "main_norm.weight": gguf.MODEL_TENSOR.ENC_OUTPUT_NORM,
        "markov_head.embed.weight": gguf.MODEL_TENSOR.DSPARK_MARKOV_W1,
        "markov_head.head.weight": gguf.MODEL_TENSOR.DSPARK_MARKOV_W2,
        "confidence_head.proj.weight": gguf.MODEL_TENSOR.DSPARK_CONF_PROJ,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_count = self.hparams["num_nextn_predict_layers"]
        if self.block_count != 3:
            raise ValueError("DeepSeek-V4.1 DSpark expects three draft stages")
        self.hparams = {
            **self.hparams,
            "num_hidden_layers": self.block_count,
            "n_routed_experts": self.hparams["dspark_n_routed_experts"],
            "num_experts_per_tok": self.hparams["dspark_num_experts_per_tok"],
            "compress_ratios": [0] * self.block_count,
            "engram_layer_ids": [],
        }
        self.tensor_map = gguf.get_tensor_name_map(self.model_arch, self.block_count)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item
        if not (match := re.fullmatch(r"mtp\.(\d+)\.(.+)", name)):
            return None
        stage, suffix = int(match[1]), match[2]
        if suffix.startswith(("main_proj.", "main_norm.")):
            if stage != 0:
                raise ValueError(f"Unexpected DSpark input projection {name}")
            return suffix, gen
        if suffix.startswith(("norm.", "markov_head.", "confidence_head.")):
            if stage != 2:
                raise ValueError(f"Unexpected DSpark output head {name}")
            return suffix, gen
        if suffix == "ffn.gate.bias_vl":
            return None
        return f"layers.{stage}.{suffix}", gen

    def _map_dsv4_tensor_name(self, name: str, bid: int | None) -> tuple[gguf.MODEL_TENSOR, str]:
        if name in self._root_map:
            return self._root_map[name], ".weight"
        return DeepseekV4Model._map_dsv4_tensor_name(self, name, bid)

    def tensor_force_quant(self, name: str, new_name: str, bid: int | None, n_dims: int) -> gguf.GGMLQuantizationType | bool:
        if name == "confidence_head.proj.weight":
            return gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name, new_name, bid, n_dims)

    def set_vocab(self):
        original_dir = self.dir_model
        try:
            self.dir_model = self.target_model_dir or self.dir_model
            super().set_vocab()
        finally:
            self.dir_model = original_dir
        self.gguf_writer.add_mask_token_id(self.hparams["dspark_noise_token_id"])

    def set_gguf_parameters(self):
        DeepseekV4Model.set_gguf_parameters(self)
        self.gguf_writer.add_block_size(self.hparams["dspark_block_size"])
        # These are target attention inputs, before the block's normalization.
        self.gguf_writer.add_target_layers(self.hparams["dspark_target_layer_ids"])
        self.gguf_writer.add_string("dflash.target_model_architecture", "deepseek41")


@ModelBase.register("DeepseekV41ForCausalLM")
class DeepseekV41VisionModel(DeepseekV4FlashVisionModel):
    model_arch = gguf.MODEL_ARCH.MMPROJ

    def get_vision_config(self) -> dict[str, Any]:
        config = self.global_config["vision_config"]
        return {**config, "image_size": config["patch_size"] * config["downsample_ratio"] * 16}

    def set_gguf_parameters(self):
        MmprojModel.set_gguf_parameters(self)
        config = self.global_config["vision_config"]
        if config.get("max_wh_ratio") is not None or config["max_image_tokens"] != 1024:
            raise ValueError("DeepSeek-V4.1 vision expects a 1024 token budget without an aspect ratio cap")
        self.gguf_writer.add_clip_projector_type(gguf.VisionProjectorType.DEEPSEEK41V)
        self.gguf_writer.add_vision_attention_layernorm_eps(1e-6)
        self.gguf_writer.add_vision_use_silu(True)
        self.gguf_writer.add_vision_projector_scale_factor(config["downsample_ratio"])
        self.gguf_writer.add_vision_min_pixels(config["min_pixels"])
