"""Opt-in runtime guard for the P3 small-op fusions (vLLM 0.31.0, SM120, Qwen3.6-27B-FP8).

Mount this directory read-only at /instrument (first on PYTHONPATH) together with f_kernels.py and, to keep
the V2 SiLU+quant candidate active, a copy of the V2 guard as v2_sitecustomize.py.

Env:
  PRO6000D_F_ENABLE=norm,conv   which fusions to install (default: none)
  PRO6000D_F_KERNELS_SHA256     sha256 of f_kernels.py (must match)
  PRO6000D_F_V2=1               also run the V2 guard (v2_sitecustomize.py; it reads its own PRO6000D_FP8_FUSION_* env)

norm : replaces QwenGatedDeltaNetAttention._output_projection's [z copy + layer_norm_fwd + group-quant] by one
       opaque op (guarded inside: M>=256, strides/dtypes as verified, SM120; otherwise the official chain runs
       inside the op).  The following FP8 GEMM is the unchanged official kernel on the identical FP8 + scales.
conv : replaces [causal_conv1d_fn -> fused_post_conv_prep] by one kernel for ONE prefill sequence with no decode /
       spec tokens (query_start_loc has 2 entries, >=4 tokens, bf16, width 4, silu, no bias, no APC); any other
       call runs the official kernels.
"""
import hashlib
import importlib.abc
import importlib.machinery
import logging
import os
import runpy
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_TARGET = "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn"
_LOGGED = set()
_ENABLED = {x for x in os.environ.get("PRO6000D_F_ENABLE", "").split(",") if x}
_LOG = logging.getLogger("pro6000d_f_fusion")
_VERIFY = os.environ.get("PRO6000D_F_VERIFY") == "1"
_VERIFY_STATS = {"calls": 0, "fp8_byte_mismatch": 0, "scale_bit_mismatch": 0}

if os.environ.get("PRO6000D_F_V2") == "1" and (_HERE / "v2_sitecustomize.py").is_file():
    runpy.run_path(str(_HERE / "v2_sitecustomize.py"))   # also runs the image's base sitecustomize
else:
    _BASE = Path("/usr/lib/python3.12/sitecustomize.py")
    if _BASE.is_file() and _BASE.resolve() != Path(__file__).resolve():
        runpy.run_path(str(_BASE))


def _note(key, msg):
    if key not in _LOGGED:
        _LOGGED.add(key)
        _LOG.warning(msg)


def _install(module):
    import torch
    sys.path.insert(0, str(_HERE))
    expected = os.environ["PRO6000D_F_KERNELS_SHA256"]
    actual = hashlib.sha256((_HERE / "f_kernels.py").read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError("f_kernels.py SHA256 mismatch")
    import f_kernels as fk

    cls = module.QwenGatedDeltaNetAttention

    # Pre-compile the Triton kernels during the engine's own warmup (attn_metadata=None profile/warmup runs call
    # _warmup_prefill_kernels) so the first real request does not pay JIT time.
    orig_warm = cls._warmup_prefill_kernels

    def warm_w(self, *a, **k):
        r = orig_warm(self, *a, **k)
        try:
            fk.prewarm(a[0].device if a else "cuda", norm="norm" in _ENABLED, conv="conv" in _ENABLED)
            _note("prewarm", "P3 kernels pre-compiled during warmup")
        except Exception as e:  # never break startup
            _note("prewarm-fail", f"P3 prewarm failed: {e!r}")
        return r

    cls._warmup_prefill_kernels = warm_w

    # ---------------------------------------------------------------- norm
    # In this image the GDN gated RMSNorm runs INSIDE the opaque custom op
    # vllm::qwen_gdn_attention_core_fused_norm_packed (_forward_core_fused_norm -> _rms_norm_gated_cuda, in place
    # on core_attn_out), and the FP8 input quant of out_proj is a separate op in the compiled graph.  We add one
    # opaque op that does the same core work but finishes with norm+quant, and call the cutlass block GEMM on its
    # FP8 + scales.  Non-fusable calls (decode, padded batches, M<256, spec-decode fused path, mixed batches)
    # run the original _forward_core_fused_norm and then the official quant op into the same q/s buffers.
    if "norm" in _ENABLED:
        from vllm.model_executor.layers.quantization.utils.quant_utils import get_fp8_min_max
        fp8_min, fp8_max = get_fp8_min_max()

        @module.eager_break_during_capture
        def pro6000d_gdn_core_norm_quant(mixed_qkvz: torch.Tensor, ba: torch.Tensor, core_attn_out: torch.Tensor,
                                         q: torch.Tensor, s: torch.Tensor, layer_name: module.LayerNameType) -> None:
            name = module._resolve_layer_name(layer_name)
            ctx = module.get_forward_context()
            self = ctx.no_compile_layers[name]
            raw = ctx.attn_metadata
            md = raw.get(self.prefix) if isinstance(raw, dict) else None
            if md is None:
                self._forward_core_fused_norm_packed(mixed_qkvz=mixed_qkvz, ba=ba, core_attn_out=core_attn_out)
                return
            qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
            mixed_qkv, gate_flat = mixed_qkvz.split([qkv_size, self.value_dim // self.tp_size], dim=-1)
            output_gate = gate_flat.reshape(gate_flat.size(0), -1, self.head_v_dim)
            b, a = self.split_ba(ba)
            M = core_attn_out.shape[0]
            spec_decode = self._can_use_fused_gdn_mtp_decode(md) and md.num_prefills == 0
            if (not spec_decode and md.num_actual_tokens == M and M >= fk.MIN_FUSED_M
                    and core_attn_out.is_contiguous() and q.shape == (M, core_attn_out.shape[1] * 128)):
                _note("norm-fused", f"P3 gdn core+norm+quant fused path used (first M={M})")
                self._forward_core(mixed_qkv=mixed_qkv, b=b.contiguous(), a=a.contiguous(), core_attn_out=core_attn_out)
                fk.gdn_norm_quant_op(q, s, core_attn_out, output_gate, self.norm.weight.contiguous(), self.norm.eps)
                if _VERIFY:
                    (qr, sr), _y = fk.gdn_norm_quant_reference(core_attn_out, output_gate, self.norm.weight.contiguous(),
                                                               self.norm.eps)
                    dq = int((q.view(torch.uint8) != qr.view(torch.uint8)).sum())
                    ds = int((s.contiguous().view(torch.int32) != sr.contiguous().view(torch.int32)).sum())
                    _VERIFY_STATS["calls"] += 1
                    _VERIFY_STATS["fp8_byte_mismatch"] += dq
                    _VERIFY_STATS["scale_bit_mismatch"] += ds
                    _LOG.warning("P3 VERIFY norm+quant in-situ: calls=%d cum_fp8_byte_mismatch=%d cum_scale_bit_mismatch=%d (M=%d)",
                                 _VERIFY_STATS["calls"], _VERIFY_STATS["fp8_byte_mismatch"],
                                 _VERIFY_STATS["scale_bit_mismatch"], M)
                return
            self._forward_core_fused_norm(mixed_qkv=mixed_qkv, b=b, a=a, output_gate=output_gate,
                                          core_attn_out=core_attn_out)
            torch.ops._C.per_token_group_fp8_quant(core_attn_out.flatten(-2), q, s, 128, 1e-10, fp8_min, fp8_max,
                                                   False, True, False)

        module.direct_register_custom_op(
            op_name="pro6000d_gdn_core_norm_quant", op_func=pro6000d_gdn_core_norm_quant,
            mutates_args=["core_attn_out", "q", "s"])

        orig_fwd = cls.forward_cuda

        def _compute_eligible(self):
            ok = False
            conds = {}
            try:
                layer = self.out_proj
                qm = layer.quant_method
                kern = getattr(qm, "fp8_linear", None)
                conds = dict(
                    qm=type(qm).__name__, kern=type(kern).__name__, tp=getattr(layer, "tp_size", None),
                    bias=getattr(layer, "bias", "missing") is None, aiq=getattr(kern, "apply_input_quant", None),
                    gs=self.norm.group_size, nbg=self.norm.norm_before_gate, act=self.norm.activation,
                    hv=self.head_v_dim, wdt=str(self.norm.weight.dtype), efd=self.enable_fused_gdn_decode)
                ok = (conds["qm"] == "Fp8LinearMethod" and conds["kern"] == "CutlassFp8BlockScaledMMKernel"
                      and conds["tp"] == 1 and conds["bias"] and conds["aiq"]
                      and conds["gs"] is None and conds["nbg"] and conds["act"] in ("swish", "silu")
                      and conds["hv"] == 128 and conds["efd"]
                      and self.norm.weight.dtype in (torch.bfloat16, torch.float32))
            except Exception as e:
                conds["exc"] = repr(e)
                ok = False
            _note("norm-elig-" + str(ok), f"P3 norm eligibility={ok} {conds}")
            self._p3_norm_ok = ok
            return ok

        orig_init = cls.__init__

        def init_w(self, *a, **k):
            orig_init(self, *a, **k)
            _compute_eligible(self)

        cls.__init__ = init_w

        def forward_cuda_w(self, hidden_states):
            if not getattr(self, "_p3_norm_ok", False) or hidden_states.dtype != torch.bfloat16:
                return orig_fwd(self, hidden_states)
            num_tokens = hidden_states.size(0)
            mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
            ba, _ = self.in_proj_ba(hidden_states)
            nh = self.num_v_heads // self.tp_size
            core_attn_out = torch.zeros((num_tokens, nh, self.head_v_dim), dtype=hidden_states.dtype,
                                        device=hidden_states.device)
            q = torch.empty((num_tokens, nh * self.head_v_dim), dtype=torch.float8_e4m3fn, device=hidden_states.device)
            s = torch.empty((nh, num_tokens), dtype=torch.float32, device=hidden_states.device).permute(1, 0)
            torch.ops.vllm.pro6000d_gdn_core_norm_quant(mixed_qkvz, ba, core_attn_out, q, s,
                                                        layer_name=module._encode_layer_name(self.prefix))
            layer = self.out_proj
            kern = layer.quant_method.fp8_linear
            bs = getattr(layer, "weight_scale_inv", None)
            if bs is None:
                bs = layer.weight_scale
            out = kern.apply_block_scaled_mm(A=q, B=layer.weight, As=s, Bs=bs)
            return out.to(dtype=kern.config.out_dtype)

        cls.forward_cuda = forward_cuda_w

    # ---------------------------------------------------------------- conv
    if "conv" in _enabled_conv():
        orig_conv = module.causal_conv1d_fn
        orig_post = module.fused_post_conv_prep
        pending = {}

        def conv_w(x, weight, bias, **kw):
            pending.clear()
            md = kw.get("metadata")
            try:
                ok = (md is not None and bias is None and x.dtype == torch.bfloat16 and x.dim() == 2
                      and x.stride(0) == 1 and x.shape[0] == 10240 and weight.shape == (10240, 4)
                      and weight.dtype == torch.bfloat16 and md.num_prefills == 1 and md.num_decodes == 0
                      and md.spec_sequence_masks is None and x.shape[1] >= 4
                      and x.shape[1] == md.num_actual_tokens
                      and kw.get("query_start_loc") is not None and kw["query_start_loc"].numel() == 2
                      and kw.get("cache_indices") is not None and kw["cache_indices"].numel() == 1
                      and kw.get("has_initial_state") is not None and kw["has_initial_state"].numel() == 1
                      and kw.get("block_idx_last_scheduled_token") is None
                      and kw.get("activation", "silu") in ("silu", "swish", True)
                      and kw["conv_states"].dtype == torch.bfloat16 and kw["conv_states"].dim() == 3
                      and kw["conv_states"].shape[1] == 10240 and kw["conv_states"].shape[2] >= 3
                      and x.device.type == "cuda" and torch.cuda.get_device_capability(x.device) == (12, 0))
            except Exception:
                ok = False
            if not ok:
                return orig_conv(x, weight, bias, **kw)
            pending.update(x=x, weight=weight, bias=bias, kw=kw, ptr=x.data_ptr())
            return x    # stand-in; consumed by post_w, which runs the fused kernel on the *input*

        def post_w(conv_output, a, b, A_log, dt_bias, num_k_heads, head_k_dim, head_v_dim,
                   apply_l2norm=True, output_g_exp=False):
            if pending:
                p = dict(pending)
                pending.clear()
                x = p["x"]
                xt = x.transpose(0, 1)
                match = (conv_output.data_ptr() == p["ptr"] and conv_output.shape == xt.shape
                         and conv_output.stride() == xt.stride() and apply_l2norm and not output_g_exp
                         and a.is_contiguous() and b.is_contiguous() and num_k_heads == 16
                         and head_k_dim == 128 and head_v_dim == 128 and A_log.shape[0] == 48
                         and a.dtype == b.dtype)
                kw = p["kw"]
                if match:
                    _note("conv-fused", f"P3 conv_post_fused kernel launched (first L={xt.shape[0]})")
                    return fk.conv_post_fused(xt, p["weight"], kw["conv_states"], kw["cache_indices"],
                                              kw["has_initial_state"], a, b, A_log, dt_bias,
                                              num_k_heads, head_k_dim, head_v_dim)
                conv_output = orig_conv(x, p["weight"], p["bias"], **kw).transpose(0, 1)
            return orig_post(conv_output, a, b, A_log, dt_bias, num_k_heads, head_k_dim, head_v_dim,
                             apply_l2norm=apply_l2norm, output_g_exp=output_g_exp)

        module.causal_conv1d_fn = conv_w
        module.fused_post_conv_prep = post_w


def _enabled_conv():
    return _ENABLED


class _AfterImportLoader(importlib.abc.Loader):
    def __init__(self, delegate, fn):
        self._delegate = delegate
        self._fn = fn

    def create_module(self, spec):
        create = getattr(self._delegate, "create_module", None)
        return create(spec) if create else None

    def exec_module(self, module):
        self._delegate.exec_module(module)
        self._fn(module)


_OP_NAME = "vllm::pro6000d_gdn_core_norm_quant"


def _patch_splitting_ops(module):
    # The new opaque op must be a graph-splitting op exactly like vllm::qwen_gdn_attention_core_fused_norm_packed;
    # otherwise the compiled pieces merge across layers and Inductor re-reads several residual tensors per norm.
    ops = module.CompilationConfig._attention_ops
    if _OP_NAME not in ops:
        ops.append(_OP_NAME)


class _Hook(importlib.abc.MetaPathFinder):
    HOOKS = {_TARGET: _install, "vllm.config.compilation": _patch_splitting_ops}

    def find_spec(self, fullname, path, target=None):
        fn = self.HOOKS.get(fullname)
        if fn is None or (fn is _patch_splitting_ops and "norm" not in _ENABLED):
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is not None and spec.loader is not None:
            spec.loader = _AfterImportLoader(spec.loader, fn)
        return spec


if _ENABLED:
    loaded = sys.modules.get("vllm.config.compilation")
    if loaded is not None and "norm" in _ENABLED:
        _patch_splitting_ops(loaded)
    loaded = sys.modules.get(_TARGET)
    if loaded is not None:
        _install(loaded)
    else:
        sys.meta_path.insert(0, _Hook())
