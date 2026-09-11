#!/usr/bin/env python3
"""Evaluate the report's CED decoder replay with the released model submodules."""

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from types import MethodType

import numpy as np
import torch
import torch.nn.functional as F

import reference


@contextmanager
def stable_index(official, enabled):
    original_forward = official.Indexer.forward
    original_topk = torch.Tensor.topk

    def topk(x, k, dim=-1, largest=True, sorted=True):
        del sorted
        ids = torch.argsort(x, dim=dim, descending=largest, stable=True).narrow(dim, 0, k)
        return torch.return_types.topk((torch.gather(x, dim, ids), ids))

    def forward(self, *args, **kwargs):
        torch.Tensor.topk = topk
        try:
            return original_forward(self, *args, **kwargs)
        finally:
            torch.Tensor.topk = original_topk

    if enabled:
        official.Indexer.forward = forward
    try:
        yield
    finally:
        official.Indexer.forward = original_forward
        torch.Tensor.topk = original_topk


def publish_decoder_cache(official, layer, hidden, pre):
    attention = layer.attn
    assert attention.is_kv_source and attention.compress_ratio == 1
    x = layer.attn_norm(layer.hc_pre(hidden, pre))
    latent = attention.compressor(x, 0)
    indexer = attention.indexer
    key = indexer.k_norm(indexer.wk(latent))
    positions = attention.freqs_cis[:hidden.size(1)]
    official.apply_rotary_emb(key[..., -attention.rope_head_dim:], positions)
    official.fp4_act_quant(key, official.fp4_block_size, True)
    indexer.k_cache[:, :key.size(1)] = key
    official.apply_rotary_emb(latent[..., -attention.rope_head_dim:], positions)
    official.fp4_act_quant(latent, 16, True, scale_dtype=torch.float8_e4m3fn)
    attention.compress_kv_cache[:, :latent.size(1)] = latent
    return x


@contextmanager
def decoder_replay(official, layers, start, end):
    saved = []
    source = layers[0].attn
    assert source.is_kv_source and source.compress_ratio == 1
    original_window = official.Attention._window_kv
    original_index = official.Indexer.forward

    def window(self, x, freqs, start_pos):
        assert start_pos == start and start_pos + x.size(1) == end
        if start == 0:
            return original_window(self, x, freqs, start_pos)
        kv = self.kv_norm(self.wkv(x))
        official.apply_rotary_emb(kv[..., -self.rope_head_dim:], freqs)
        official.act_quant(kv, official.fp8_block_size, official.scale_fmt, official.scale_dtype, True)
        positions = torch.arange(start, end, device=x.device)
        self.window_kv_cache[:x.size(0), positions % self.window_size] = kv
        slots = positions[:, None] - self.window_size + 1 + torch.arange(self.window_size, device=x.device)
        ids = torch.where(slots >= start, slots - start, -1)
        ids = ids.int().unsqueeze(0).expand(x.size(0), -1, -1).contiguous()
        return kv, ids

    def index(self, x, qr, latent, start_pos, offset):
        assert latent is None and start_pos == start
        if start == 0:
            return original_index(self, x, qr, latent, start_pos, offset)
        candidates = official.shared_attn.candidates
        selected, masks = [], []
        width = min(self.index_topk, end)
        for token in range(x.size(1)):
            pos = start + token
            if self.uses_candidates:
                official.shared_attn.candidates = candidates[:, token:token + 1, :pos + 1]
            ids = original_index(self, x[:, token:token + 1], qr[:, token:token + 1], None, pos, offset)
            selected.append(F.pad(ids, (0, width - ids.size(-1)), value=-1))
            if self.is_candidate_source:
                masks.append(F.pad(official.shared_attn.candidates, (0, end - pos - 1), value=False))
        official.shared_attn.candidates = torch.cat(masks, dim=1) if masks else candidates
        return torch.cat(selected, dim=1)

    def compressed(self, x, qr, start_pos, offset):
        assert self.compress_ratio == 1 and start_pos == start
        official.shared_attn.compress_kv = source.compress_kv_cache
        official.shared_attn.index_k = source.indexer.k_cache
        ids = self._compress_topk_idxs(x, qr, None, start_pos, offset, end)
        return source.compress_kv_cache[:x.size(0), :end], ids

    try:
        for layer in layers:
            attention = layer.attn
            assert layer.engram is None
            for name, method in [('_window_kv', window), ('_compress_kv', compressed)]:
                saved.append((attention, name, name in attention.__dict__, attention.__dict__.get(name)))
                setattr(attention, name, MethodType(method, attention))
            if attention.indexer is not None:
                saved.append((attention.indexer, 'forward', 'forward' in attention.indexer.__dict__, attention.indexer.__dict__.get('forward')))
                attention.indexer.forward = MethodType(index, attention.indexer)
        yield
    finally:
        for module, name, present, method in reversed(saved):
            if present:
                setattr(module, name, method)
            else:
                delattr(module, name)


@torch.inference_mode()
def prefill(model, official, tokens, replay_tokens):
    assert len(model.layers) == 40 and replay_tokens > 0
    split = 20
    end = tokens.size(1)
    start = max(0, end - replay_tokens)
    assert end - start <= model.layers[split].attn.window_size or start == 0
    hashes = model.engram_hash(tokens, 0) if model.engram_hash is not None else None
    hidden = model.embed(tokens).unsqueeze(2).repeat(1, 1, model.hc_mult, 1)
    pre = official.make_identity_pre_mix(hidden, model.hc_mult)
    for layer in model.layers[:split]:
        if layer.engram is not None:
            hidden = layer.engram(hidden, hashes[:, :, layer.engram.layer_hash_index, :])
        hidden, pre = layer(hidden, 0, pre, None)
    frontier = hidden.clone(), pre.clone()
    boundary_input = publish_decoder_cache(official, model.layers[split], hidden, pre)
    source = model.layers[split].attn
    published = source.compress_kv_cache.clone(), source.indexer.k_cache.clone()
    hidden, pre = hidden[:, start:].contiguous(), pre[:, start:].contiguous()
    with decoder_replay(official, model.layers[split:], start, end):
        for layer in model.layers[split:]:
            hidden, pre = layer(hidden, start, pre, None)
    torch.testing.assert_close(source.compress_kv_cache, published[0], rtol=0, atol=0)
    torch.testing.assert_close(source.indexer.k_cache, published[1], rtol=0, atol=0)
    logits = model.head(model.norm(model.layers[-1].hc_pre(hidden, pre)))
    return logits, frontier, boundary_input, start


@torch.inference_mode()
def run(args):
    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(args.threads)
    source = args.source.resolve()
    official = reference.load_official(source)
    config = reference.fixture_args(official, args.prompt_length + args.decode_tokens + 16, args.dense_type, args.expert_type)
    model = official.Transformer(config, reference.FixtureTokenizer(config.vocab_size)).eval()
    reference.initialize(model, official, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    reference.save_weights(model, args.output)
    tokens = torch.randint(3, config.vocab_size, (1, args.prompt_length + args.decode_tokens), generator=torch.Generator().manual_seed(args.seed + 1))
    stages, handles = reference.capture(model)
    counts = {str(layer): [] for layer in range(config.n_layers)}
    for layer in model.layers:
        def record_count(module, inputs):
            counts[str(module.layer_id)].append(inputs[0].size(1))
        handles.append(layer.register_forward_pre_hook(record_count))
        if layer.attn.indexer is not None and layer.attn.indexer.owns_k:
            def bind_owner(module, inputs):
                del inputs
                official.shared_attn.index_k = module.k_cache
            handles.append(layer.attn.indexer.register_forward_pre_hook(bind_owner))
    reports = []
    try:
        with stable_index(official, args.stable_index):
            logits, frontier, boundary_input, start = prefill(model, official, tokens[:, :args.prompt_length], args.replay_tokens or config.window_size)
            arrays = dict(stages)
            arrays.update(reference.cache_arrays(model, args.prompt_length))
            arrays.update(logits=logits.float().numpy(), tokens=tokens[:, :args.prompt_length].numpy(), encoder_hidden=frontier[0].float().numpy(), encoder_pre=frontier[1].float().numpy(), decoder_global_input=boundary_input.float().numpy())
            np.savez(args.output / 'stages-00000.npz', **arrays)
            reports.append(dict(start=0, end=args.prompt_length, decoder_start=start, file='stages-00000.npz', selected_token=int(logits.argmax(-1).item())))
            for pos in range(args.prompt_length, tokens.size(1)):
                stages.clear()
                selected, logits, _ = model(tokens[:, pos:pos + 1], pos)
                arrays = dict(stages)
                arrays.update(reference.cache_arrays(model, pos + 1))
                arrays.update(logits=logits.float().numpy(), tokens=tokens[:, pos:pos + 1].numpy())
                filename = f'stages-{pos:05d}.npz'
                np.savez(args.output / filename, **arrays)
                reports.append(dict(start=pos, end=pos + 1, file=filename, selected_token=int(selected.item())))
    finally:
        for handle in handles:
            handle.remove()
    for report in reports:
        with np.load(args.output / report['file']) as arrays:
            for name, value in arrays.items():
                assert not np.isnan(value).any() if name.endswith('score_state') else np.isfinite(value).all(), name
    for layer in range(config.n_layers):
        prefill_count = args.prompt_length if layer < 20 else args.prompt_length - start
        assert counts[str(layer)] == [prefill_count] + [1]*args.decode_tokens
    report = dict(
        source=str(source), source_sha256={name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in ('model.py', 'kernel.py', 'engram.py')},
        adapter_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ('reference.py', 'cpu_kernel.py', 'ced-reference.py')},
        config=asdict(config), seed=args.seed, torch_version=torch.__version__, numpy_version=np.__version__, threads=args.threads,
        execution='Official submodules with CPU TileLang translations and explicit CED decoder replay from technical report sections 2.2 and 3.2.2.',
        limitation='Bounded replay is an intentional approximation. This is not released CUDA execution or a model-quality result. The reference retains the full encoder frontier for inspection.',
        index_ties='ascending original ID' if args.stable_index else 'PyTorch topk',
        errata=['Bind each index-key owner even when its compressor returns no new latent.'],
        decoder_global_cache_unchanged_during_replay=True, layer_token_counts=counts, runs=reports,
    )
    (args.output / 'reference.json').write_text(json.dumps(report, indent=2) + '\n')
    (args.output / 'tokens.txt').write_text(' '.join(map(str, tokens.flatten().tolist())) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prompt-length', type=int, default=193)
    parser.add_argument('--decode-tokens', type=int, default=9)
    parser.add_argument('--replay-tokens', type=int, default=0)
    parser.add_argument('--seed', type=int, default=41)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--dense-type', choices=['fp8', 'bf16'], default='fp8')
    parser.add_argument('--expert-type', choices=['fp4', 'bf16'], default='fp4')
    parser.add_argument('--stable-index', action='store_true')
    args = parser.parse_args()
    assert args.prompt_length > 0 and args.decode_tokens >= 0 and args.replay_tokens >= 0
    run(args)
