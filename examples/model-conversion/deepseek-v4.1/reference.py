#!/usr/bin/env python3
"""Run the official V4.1 Python graph with small weights and CPU kernel definitions."""

import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

import cpu_kernel


class FixtureTokenizer:
    def __init__(self, size):
        self.size = size
        self.backend_tokenizer = self

    def __len__(self):
        return self.size

    def decode(self, ids, skip_special_tokens=False):
        del skip_special_tokens
        return ''.join(self.id_to_token(index) for index in ids)

    def id_to_token(self, index):
        return f'token{index:04d}'


def load_official(source):
    previous = sys.modules.get('kernel')
    sys.modules['kernel'] = cpu_kernel
    sys.path.insert(0, str(source))
    try:
        spec = importlib.util.spec_from_file_location('dsv41_official_model', source / 'model.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)
        if previous is None:
            del sys.modules['kernel']
        else:
            sys.modules['kernel'] = previous


def fixture_args(official, context, dense_type, expert_type):
    args = official.ModelArgs(
        max_batch_size=1, max_seq_len=context, temperature=0,
        dtype=dense_type, expert_dtype='fp4' if expert_type == 'fp4' else None,
        vocab_size=128, dim=64, moe_inter_dim=64, n_layers=40, n_mtp_layers=0,
        n_heads=4, n_routed_experts=8, n_shared_experts=1, n_activated_experts=2,
        route_scale=1.5, swiglu_limit=10.0, q_lora_rank=64, head_dim=64, rope_head_dim=32,
        norm_eps=1e-20, o_groups=2, o_lora_rank=32, window_size=16,
        compress_ratios=(0, 0) + (2,) * 18 + (1,) * 20,
        kv_source_layers=(2, 8, 14, 20), index_source_layers=(2, 8, 14, 20, 24, 28, 32, 36),
        compress_rope_theta=160000, original_seq_len=65536, rope_theta=10000, rope_factor=16,
        beta_fast=32, beta_slow=1, index_n_heads=4, index_head_dim=64, index_topk=8,
        candidate_source_layer=20, candidate_topk_blocks=4, candidate_block_size=4,
        hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6,
        engram_layer_ids=(1, 14), engram_num_embeddings=(0, 0), engram_max_ngram_size=4,
        engram_vocab_size=17, engram_n_heads=2, engram_head_dim=32, engram_pad_id=2,
        engram_compressed_vocab_size=128, vision_n_layers=0,
    )
    layout = official.EngramLayout.from_args(args)
    args.engram_num_embeddings = tuple(sum(sum(group) for group in layer) for layer in layout.primes)
    return args


@torch.no_grad()
def initialize(model, official, seed):
    generator = torch.Generator().manual_seed(seed)

    def random(shape):
        return torch.randn(shape, generator=generator, dtype=torch.float32)

    initialized = set()
    for module in model.modules():
        if isinstance(module, official.Linear):
            values = random((module.out_features, module.in_features)) * (0.35 / math.sqrt(module.in_features))
            if module.weight.dtype == torch.float4_e2m1fn_x2:
                packed, scales = cpu_kernel.fp4_act_quant(values, 32)
                module.weight.view(torch.uint8).copy_(packed.view(torch.uint8))
                module.scale.copy_(scales)
            elif module.weight.dtype == torch.float8_e4m3fn:
                padded = F.pad(values, (0, -values.shape[1] % 32, 0, -values.shape[0] % 32))
                blocks = padded.unflatten(0, (-1, 32)).unflatten(-1, (-1, 32))
                scales = cpu_kernel.round_scale(blocks.abs().amax((1, 3)).clamp_min(1e-4) * (1 / 448.0))
                expanded = scales.repeat_interleave(32, 0).repeat_interleave(32, 1)
                module.weight.copy_((values / expanded[:values.shape[0], :values.shape[1]]).to(module.weight.dtype))
                module.scale.copy_(scales)
            else:
                module.weight.copy_(values)
            initialized.add(id(module.weight))
            if module.scale is not None:
                initialized.add(id(module.scale))
        elif isinstance(module, official.ParallelEngramEmbedding):
            values = random(module.weight.shape) * 0.1
            packed, scales = cpu_kernel.act_quant(values, 32, 'ue8m0', torch.float8_e8m0fnu)
            module.weight.copy_(packed)
            module.scale.copy_(scales)
            initialized.update((id(module.weight), id(module.scale)))

    for name, parameter in model.named_parameters():
        if id(parameter) in initialized:
            continue
        assert parameter.dtype in (torch.float32, torch.bfloat16), (name, parameter.dtype)
        values = random(parameter.shape)
        if 'norm.weight' in name or name.endswith(('q_weight', 'k_weight')):
            values = 1 + values * 0.05
        elif name.endswith(('_scale',)):
            values = 0.3 + values * 0.03
        elif name.endswith(('_fn',)):
            values *= 0.2 / math.sqrt(parameter.shape[-1])
        elif name.endswith(('_base',)):
            values *= 0.1
        elif name.endswith(('bias', 'bias_vl')):
            values *= 0.01
        elif name.endswith('attn_sink'):
            values = -0.5 + values * 0.1
        elif name == 'embed.weight':
            values *= 0.5
        else:
            values *= 0.35 / math.sqrt(parameter.shape[-1])
        parameter.copy_(values)


def save_weights(model, output):
    arrays, metadata = {}, {}
    for name, parameter in model.named_parameters():
        value = parameter.detach().contiguous()
        metadata[name] = dict(dtype=str(value.dtype), shape=list(value.shape))
        if value.dtype == torch.bfloat16:
            value = value.view(torch.uint16)
        elif value.dtype != torch.float32:
            value = value.view(torch.uint8)
        arrays[name] = value.numpy()
    np.savez(output / 'weights.npz', **arrays)
    (output / 'weights.json').write_text(json.dumps(metadata, indent=2) + '\n')


def capture(model):
    arrays, handles = {}, []

    def save(name, tensor):
        arrays[name] = tensor.detach().float().numpy().copy() if tensor.is_floating_point() else tensor.detach().numpy().copy()

    def hook(name):
        def record(module, inputs, result):
            del module, inputs
            if isinstance(result, tuple):
                for index, tensor in enumerate(result):
                    if tensor is not None:
                        save(f'{name}.{index}', tensor)
            elif result is not None:
                save(name, result)
        return record

    for name, module in model.named_modules():
        if name.startswith('layers.') and (name.count('.') == 1 or name.endswith((
            '.attn', '.attn_norm', '.ffn', '.ffn_norm', '.ffn.gate', '.engram',
            '.attn.wq_a', '.attn.wq_b', '.attn.q_norm', '.attn.kv_norm',
            '.attn.compressor', '.attn.indexer',
        ))):
            handles.append(module.register_forward_hook(hook(name)))
    return arrays, handles


def cache_arrays(model, end_pos):
    arrays = {}
    for layer in model.layers:
        attention = layer.attn
        name = f'layers.{layer.layer_id}.attn'
        arrays[name + '.window_kv_cache'] = attention.window_kv_cache.float().numpy().copy()
        if attention.is_kv_source:
            length = end_pos // attention.compress_ratio
            arrays[name + '.compress_kv_cache'] = attention.compress_kv_cache[:, :length].float().numpy().copy()
            arrays[name + '.indexer.k_cache'] = attention.indexer.k_cache[:, :length].float().numpy().copy()
            if attention.compress_ratio > 1:
                arrays[name + '.compressor.kv_state'] = attention.compressor.kv_state.numpy().copy()
                arrays[name + '.compressor.score_state'] = attention.compressor.score_state.numpy().copy()
    arrays['engram_hash.cache'] = model.engram_hash.cache[:, :end_pos].numpy().copy()
    return arrays


@torch.inference_mode()
def run(args):
    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(args.threads)
    source = args.source.resolve()
    official = load_official(source)
    config = fixture_args(official, args.prompt_length + args.decode_tokens + 16, args.dense_type, args.expert_type)
    model = official.Transformer(config, FixtureTokenizer(config.vocab_size)).eval()
    initialize(model, official, args.seed)
    owner_hooks = []
    if args.bind_index_cache_owner:
        def bind_owner(module, inputs):
            del inputs
            official.shared_attn.index_k = module.k_cache

        for layer in model.layers:
            indexer = layer.attn.indexer
            if indexer is not None and indexer.owns_k:
                owner_hooks.append(indexer.register_forward_pre_hook(bind_owner))
    args.output.mkdir(parents=True, exist_ok=True)
    save_weights(model, args.output)
    generator = torch.Generator().manual_seed(args.seed + 1)
    tokens = torch.randint(3, config.vocab_size, (1, args.prompt_length + args.decode_tokens), generator=generator)
    stages, handles = capture(model)
    runs = [(0, args.prompt_length)] + [(position, position + 1) for position in range(args.prompt_length, tokens.shape[1])]
    run_reports = []
    for start, end in runs:
        stages.clear()
        selected, logits, _ = model(tokens[:, start:end], start)
        arrays = dict(stages)
        arrays.update(cache_arrays(model, end))
        arrays['tokens'] = tokens[:, start:end].numpy()
        arrays['logits'] = logits.float().numpy()
        for name, value in arrays.items():
            if name.endswith('score_state'):
                assert not np.isnan(value).any(), name
            else:
                assert np.isfinite(value).all(), name
        for layer in range(config.n_layers):
            ids = arrays[f'layers.{layer}.ffn.gate.1']
            assert ((ids >= 0) & (ids < config.n_routed_experts)).all()
            assert (np.diff(np.sort(ids, axis=-1), axis=-1) > 0).all()
            np.testing.assert_allclose(arrays[f'layers.{layer}.ffn.gate.0'].sum(-1), config.route_scale, rtol=1e-6)
        filename = f'stages-{start:05d}.npz'
        np.savez(args.output / filename, **arrays)
        run_reports.append(dict(start=start, end=end, arrays=len(arrays), selected_token=int(selected.item()), file=filename))
    for handle in handles + owner_hooks:
        handle.remove()
    references = {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in ('model.py', 'kernel.py', 'engram.py')}
    report = dict(
        source=str(source), source_sha256=references, config=asdict(config), seed=args.seed,
        adapter_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ('reference.py', 'cpu_kernel.py')},
        torch_version=torch.__version__, numpy_version=np.__version__, threads=args.threads,
        execution='official Python model with CPU replacements for six TileLang kernels',
        limitation='CPU reductions are not CUDA instruction emulation; CUDA reference parity remains unverified.',
        errata=['Bind each index-key owner even when its compressor returns no new latent.'] if args.bind_index_cache_owner else [],
        runs=run_reports,
    )
    (args.output / 'reference.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(output=str(args.output), runs=run_reports), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Official inference directory containing model.py and kernel.py')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prompt-length', type=int, default=33)
    parser.add_argument('--decode-tokens', type=int, default=4)
    parser.add_argument('--dense-type', choices=('bf16', 'fp8'), default='fp8')
    parser.add_argument('--expert-type', choices=('bf16', 'fp4'), default='fp4')
    parser.add_argument('--seed', type=int, default=41)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--bind-index-cache-owner', action='store_true', help='Apply the recorded incomplete-group cache-owner correction through a forward hook')
    arguments = parser.parse_args()
    if arguments.prompt_length < 1 or arguments.decode_tokens < 0 or arguments.threads < 1:
        parser.error('prompt length and threads must be positive; decode tokens must be nonnegative')
    if arguments.expert_type == 'bf16' and arguments.dense_type != 'bf16':
        parser.error('BF16 experts require --dense-type bf16 in the official model')
    run(arguments)
