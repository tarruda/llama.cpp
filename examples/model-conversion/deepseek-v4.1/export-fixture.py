#!/usr/bin/env python3
"""Export reference.py's small weights to GGUF without changing decoded values."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'gguf-py'))
import gguf

import cpu_kernel
import reference


def tensor_name(name):
    globals_ = {'embed.weight': 'token_embd.weight', 'norm.weight': 'output_norm.weight', 'head.weight': 'output.weight'}
    if name in globals_:
        return globals_[name]
    match = re.fullmatch(r'layers\.(\d+)\.(.+)', name)
    assert match, name
    layer, suffix = match.groups()
    if suffix.startswith('hc_'):
        return f'blk.{layer}.{suffix}.weight'
    names = {
        'attn.attn_sink': 'attn_sinks.weight',
        'attn.wq_a.weight': 'attn_q_a.weight', 'attn.wq_b.weight': 'attn_q_b.weight',
        'attn.q_norm.weight': 'attn_q_a_norm.weight', 'attn.kv_norm.weight': 'attn_kv_a_norm.weight',
        'attn.wkv.weight': 'attn_kv.weight', 'attn.wo_a.weight': 'attn_output_a.weight', 'attn.wo_b.weight': 'attn_output_b.weight',
        'attn.compressor.wkv.weight': 'attn_compressor_kv.weight', 'attn.compressor.wgate.weight': 'attn_compressor_gate.weight',
        'attn.compressor.norm.weight': 'attn_compressor_norm.weight',
        'attn.indexer.wq_b.weight': 'indexer.attn_q_b.weight', 'attn.indexer.wk.weight': 'indexer.attn_k.weight',
        'attn.indexer.k_norm.weight': 'indexer.k_norm.weight', 'attn.indexer.weights_proj.weight': 'indexer.proj.weight',
        'attn_norm.weight': 'attn_norm.weight', 'ffn_norm.weight': 'ffn_norm.weight',
        'ffn.gate.weight': 'ffn_gate_inp.weight', 'ffn.gate.bias': 'exp_probs_b.bias', 'ffn.gate.bias_vl': 'exp_probs_b_vl.bias',
        'ffn.shared_experts.w1.weight': 'ffn_gate_shexp.weight', 'ffn.shared_experts.w2.weight': 'ffn_down_shexp.weight', 'ffn.shared_experts.w3.weight': 'ffn_up_shexp.weight',
        'engram.embed.weight': 'engram_embd.weight', 'engram.embed.scale': 'engram_embd_scale.weight',
        'engram.wkv.weight': 'engram_kv.weight', 'engram.k_weight': 'engram_k_weight.weight', 'engram.q_weight': 'engram_q_weight.weight',
    }
    return f'blk.{layer}.{names[suffix]}'


def write_metadata(writer, config, official):
    integers = {
        'block_count': 'n_layers', 'embedding_length': 'dim', 'context_length': 'max_seq_len',
        'attention.head_count': 'n_heads', 'attention.key_length': 'head_dim', 'attention.value_length': 'head_dim',
        'attention.q_lora_rank': 'q_lora_rank', 'attention.sliding_window': 'window_size',
        'attention.output_group_count': 'o_groups', 'attention.output_lora_rank': 'o_lora_rank',
        'expert_count': 'n_routed_experts', 'expert_used_count': 'n_activated_experts', 'expert_shared_count': 'n_shared_experts',
        'expert_feed_forward_length': 'moe_inter_dim', 'rope.dimension_count': 'rope_head_dim', 'rope.scaling.original_context_length': 'original_seq_len',
        'attention.indexer.head_count': 'index_n_heads', 'attention.indexer.key_length': 'index_head_dim', 'attention.indexer.top_k': 'index_topk',
        'attention.indexer.candidate_source_layer': 'candidate_source_layer', 'attention.indexer.candidate_topk_blocks': 'candidate_topk_blocks', 'attention.indexer.candidate_block_size': 'candidate_block_size',
        'hyper_connection.count': 'hc_mult', 'hyper_connection.sinkhorn_iterations': 'hc_sinkhorn_iters',
        'engram.ngram_size': 'engram_max_ngram_size', 'engram.head_count': 'engram_n_heads', 'engram.head_dim': 'engram_head_dim',
        'engram.compressed_vocab_size': 'engram_compressed_vocab_size', 'engram.pad_token_id': 'engram_pad_id',
    }
    floats = {
        'attention.layer_norm_rms_epsilon': 'norm_eps', 'expert_weights_scale': 'route_scale', 'hyper_connection.epsilon': 'hc_eps',
        'rope.freq_base': 'rope_theta', 'rope.scaling.factor': 'rope_factor', 'rope.scaling.yarn_beta_fast': 'beta_fast', 'rope.scaling.yarn_beta_slow': 'beta_slow',
        'attention.compress_rope_freq_base': 'compress_rope_theta',
    }
    arrays = {
        'attention.compress_ratios': 'compress_ratios', 'attention.kv_source_layers': 'kv_source_layers', 'attention.index_source_layers': 'index_source_layers',
        'engram.layers': 'engram_layer_ids', 'engram.embedding_counts': 'engram_num_embeddings',
    }
    for key, field in integers.items():
        writer.add_uint32('deepseek41.' + key, config[field] & 0xffffffff)
    for key, field in floats.items():
        writer.add_float32('deepseek41.' + key, float(config[field]))
    for key, field in arrays.items():
        writer.add_key_value('deepseek41.' + key, config[field], gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.UINT32)
    for key in ('swiglu_clamp_exp', 'swiglu_clamp_shexp'):
        writer.add_array('deepseek41.' + key, [float(config['swiglu_limit'])] * config['n_layers'])
    writer.add_uint32('deepseek41.attention.head_count_kv', 1)
    writer.add_uint32('deepseek41.expert_gating_func', 4)
    writer.add_bool('deepseek41.expert_weights_norm', config['norm_topk_prob'])
    writer.add_string('deepseek41.rope.scaling.type', 'yarn')
    writer.add_string('deepseek41.dense_activation_dtype', config['dtype'])
    writer.add_string('deepseek41.expert_activation_dtype', 'fp8' if config['expert_dtype'] == 'fp4' else config['dtype'])
    args = official.ModelArgs(**config)
    hash_state = official.NgramHashState(args, official.EngramLayout.from_args(args), reference.FixtureTokenizer(config['vocab_size']))
    for key, tensor in {'head_bucket_sizes': hash_state.primes, 'head_offsets': hash_state.offsets, 'multipliers': hash_state.multipliers, 'token_map': hash_state.token_map}.items():
        element_type = gguf.GGUFValueType.UINT64 if key == 'multipliers' else gguf.GGUFValueType.UINT32
        writer.add_key_value('deepseek41.engram.' + key, tensor.flatten().tolist(), gguf.GGUFValueType.ARRAY, element_type)
    writer.add_tokenizer_model('llama')
    writer.add_token_list([f'token{i:04d}' for i in range(config['vocab_size'])])
    writer.add_token_types([2, 3, 3] + [1] * (config['vocab_size'] - 3))
    writer.add_token_scores([0.0] * config['vocab_size'])
    writer.add_unk_token_id(0)
    writer.add_bos_token_id(1)
    writer.add_eos_token_id(2)


def run(folder, output, keep_expert_fp4=False):
    report = json.loads((folder / 'reference.json').read_text())
    metadata = json.loads((folder / 'weights.json').read_text())
    source = Path(report['source'])
    for name, expected in report['source_sha256'].items():
        assert hashlib.sha256((source / name).read_bytes()).hexdigest() == expected, name
    official = reference.load_official(source)
    writer = gguf.GGUFWriter(output, 'deepseek41')
    writer.add_name('DeepSeek V4.1 numerical fixture')
    write_metadata(writer, report['config'], official)
    weights = np.load(folder / 'weights.npz')
    consumed = set()

    def decode(name):
        data, dtype = weights[name], metadata[name]['dtype']
        consumed.add(name)
        if '.engram.embed.' in name:
            return data.view(np.int8), gguf.GGMLQuantizationType.I8
        if dtype == 'torch.bfloat16':
            return data, gguf.GGMLQuantizationType.BF16
        if dtype == 'torch.float32':
            return data, gguf.GGMLQuantizationType.F32
        scale_name = name.removesuffix('.weight') + '.scale'
        scales = torch.from_numpy(weights[scale_name]).view(torch.float8_e8m0fnu).float()
        consumed.add(scale_name)
        if dtype == 'torch.float4_e2m1fn_x2':
            values = cpu_kernel.unpack_fp4(torch.from_numpy(data))
            expanded = scales.repeat_interleave(32, 1)
            if keep_expert_fp4:
                blocks = data.reshape(data.shape[0], -1, 16)
                codes = np.stack((blocks & 15, blocks >> 4), axis=-1).reshape(*blocks.shape[:-1], 32)
                packed = np.concatenate((weights[scale_name][..., None], codes[..., :16] | (codes[..., 16:] << 4)), axis=-1)
                packed = packed.reshape(data.shape[0], -1)
                np.testing.assert_array_equal(gguf.dequantize(packed, gguf.GGMLQuantizationType.MXFP4), (values * expanded).numpy())
                return packed, gguf.GGMLQuantizationType.MXFP4
        else:
            assert dtype == 'torch.float8_e4m3fn', (name, dtype)
            values = torch.from_numpy(data).view(torch.float8_e4m3fn).float()
            expanded = scales.repeat_interleave(32, 0).repeat_interleave(32, 1)
        return (values * expanded[:values.shape[0], :values.shape[1]]).numpy(), gguf.GGMLQuantizationType.F32

    for name in weights.files:
        if name in consumed or (name.endswith('.scale') and '.engram.embed.' not in name):
            continue
        expert = re.fullmatch(r'layers\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.weight', name)
        if expert:
            layer, _, projection = expert.groups()
            pieces = [decode(f'layers.{layer}.ffn.experts.{eid}.{projection}.weight') for eid in range(report['config']['n_routed_experts'])]
            assert all(dtype == pieces[0][1] for _, dtype in pieces)
            data, dtype = np.stack([data for data, _ in pieces]), pieces[0][1]
            mapped = f'blk.{layer}.ffn_' + {'w1': 'gate', 'w2': 'down', 'w3': 'up'}[projection] + '_exps.weight'
        else:
            data, dtype = decode(name)
            mapped = tensor_name(name)
        writer.add_tensor(mapped, np.ascontiguousarray(data), raw_dtype=dtype)
    assert consumed == set(weights.files), sorted(set(weights.files) - consumed)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(json.dumps(dict(output=str(output), source_parameters=len(consumed), bytes=output.stat().st_size, keep_expert_fp4=keep_expert_fp4)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--keep-expert-fp4', action='store_true', help='Keep source FP4 weights as MXFP4 to exercise native expert kernels')
    args = parser.parse_args()
    run(args.reference, args.output, args.keep_expert_fp4)
