#!/usr/bin/env python3
"""Root-only isolated CUDA smoke: locked ABI1 branch, fallback and Graph."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if os.environ.get('PRO6000D_MAINLOOP_MODE') != 'candidate':
        raise RuntimeError('smoke requires explicit candidate mode')
    from candidate_abi import CandidateABI, LIBRARY_SHA, TARGET_M
    if (os.environ.get('PRO6000D_MAINLOOP_LIBRARY_SHA256') != LIBRARY_SHA or
        os.environ.get('PRO6000D_MAINLOOP_VALIDATED_LIBRARY_SHA256') != LIBRARY_SHA):
        raise RuntimeError('smoke requires the locked validated SHA')
    import torch
    import vllm.compilation.passes.fusion.act_quant_fusion  # noqa: F401
    from candidate_op import last_dispatch_for_eager_smoke, register
    from packing import lookup_packed, prepare_model
    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError('SM120 required')
    abi = CandidateABI(os.environ['PRO6000D_MAINLOOP_LIBRARY'], LIBRARY_SHA)
    register(abi)
    torch.manual_seed(23062026)
    device = 'cuda'
    class CutlassFp8BlockScaledMMKernel: pass
    class Fp8LinearMethod:
        def __init__(self): self.fp8_linear = CutlassFp8BlockScaledMMKernel()
    class MergedColumnParallelLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.quant_method = Fp8LinearMethod()
            self.weight = torch.nn.Parameter(torch.randn((34816, 5120), device=device).to(torch.float8_e4m3fn), requires_grad=False)
            self.register_buffer('weight_scale_inv', (0.01 + torch.rand((272, 40), device=device)).float())
    class Root(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = torch.nn.Module()
            self.language_model.layers = torch.nn.ModuleList([torch.nn.Module()])
            self.language_model.layers[0].mlp = torch.nn.Module()
            self.language_model.layers[0].mlp.gate_up_proj = MergedColumnParallelLinear()
    model = Root()
    prepare_model(model, str(args.output.parent / 'startup-pack-smoke'))
    layer = model.language_model.layers[0].mlp.gate_up_proj
    w, ws = layer.weight, layer.weight_scale_inv
    if lookup_packed(w, ws) is None: raise RuntimeError('startup identity registration failed')
    def inputs(m):
        return (torch.randn((m, 5120), device=device).to(torch.float8_e4m3fn),
                (0.01 + torch.rand((40, m), device=device)).float().t())
    def ref(a, asc, b, bs):
        d = torch.empty((a.shape[0], 34816), dtype=torch.bfloat16, device=device)
        q = torch.empty((a.shape[0], 17408), dtype=torch.float8_e4m3fn, device=device)
        sc = torch.empty((136, a.shape[0]), dtype=torch.float32, device=device).t()
        torch.ops._C.cutlass_scaled_mm(d, a, b.t(), asc, bs.t(), None)
        torch.ops.pro6000d_fp8.silu_and_mul_per_block_quant(q, d, sc, 128, None, True)
        return q, sc
    def compare(q, sc, qr, sr):
        return {'fp8_mismatches': int((q.view(torch.uint8) != qr.view(torch.uint8)).sum().item()),
                'scale_mismatches': int((sc.view(torch.int32) != sr.view(torch.int32)).sum().item()),
                'scale_stride': list(sc.stride())}
    def case(label, m, registered, expected):
        b, bs = (w, ws) if registered else (w.clone(), ws.clone())
        if not registered and lookup_packed(b, bs) is not None: raise RuntimeError('unregistered tensor unexpectedly authorized')
        a, asc = inputs(m)
        qr, sr = ref(a, asc, b, bs)
        q, sc = torch.ops.pro6000d_mainloop_gateup.fused(a, b, asc, bs)
        torch.cuda.synchronize()
        check = compare(q, sc, qr, sr)
        row = {'case': label, 'M': m, 'dispatch': last_dispatch_for_eager_smoke(), 'dispatch_expected': expected, **check}
        row['pass'] = row['dispatch'] == expected and check['fp8_mismatches'] == check['scale_mismatches'] == 0 and check['scale_stride'] == [1, m]
        return row
    target_dispatch = 'mainloop_warp24n64_v3_m16160_r1_s1'
    fallback = 'canonical_v2_fallback'
    cases = [case('registered_target', TARGET_M, True, target_dispatch),
             case('unregistered_target', TARGET_M, False, fallback)]
    for m in (1, 5, 256, 2741, 32768):
        cases.append(case('registered_fallback', m, True, fallback))
        torch.cuda.empty_cache()
    # Graph returns fresh output buffers; references and replay stay outside capture.
    a, asc = inputs(TARGET_M)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            q, sc = torch.ops.pro6000d_mainloop_gateup.fused(a, w, asc, ws)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        qg, sg = torch.ops.pro6000d_mainloop_gateup.fused(a, w, asc, ws)
    graph.replay()
    qr, sr = ref(a, asc, w, ws)
    torch.cuda.synchronize()
    capture_check = compare(qg, sg, qr, sr)
    previous = qg.view(torch.uint8).clone()
    original_a, original_sc = a.view(torch.uint8).clone(), asc.clone()
    next_a, next_sc = inputs(TARGET_M)
    a.copy_(next_a); asc.copy_(next_sc)
    graph.replay()
    qr, sr = ref(a, asc, w, ws)
    torch.cuda.synchronize()
    replay_check = compare(qg, sg, qr, sr)
    changed = {'input_bytes_changed': not torch.equal(original_a, a.view(torch.uint8)),
               'input_scales_changed': not torch.equal(original_sc.view(torch.int32), asc.view(torch.int32)),
               'output_changed': not torch.equal(previous, qg.view(torch.uint8))}
    graph_pass = all(changed.values()) and all(c['fp8_mismatches'] == c['scale_mismatches'] == 0 for c in (capture_check, replay_check))
    report = {'scope': 'synthetic CUDA op/identity/Graph smoke, no model or API timing',
              'library_sha256': LIBRARY_SHA, 'abi_version': 1, 'variant': 'warp24n64_v3',
              'target_M': TARGET_M, 'raster': 1, 'swizzle': 1, 'library_metadata': abi.metadata,
              'cases': cases, 'graph': {'capture': capture_check, 'changed_input_replay': replay_check, **changed, 'pass': graph_pass},
              'passed': all(c['pass'] for c in cases) and graph_pass}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    if not report['passed']:raise SystemExit(1)


if __name__ == '__main__':main()
