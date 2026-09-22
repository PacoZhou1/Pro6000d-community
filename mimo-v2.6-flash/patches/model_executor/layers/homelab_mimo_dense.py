"""Opt-in, bounded MiMo TP4 BF16 projection trial for RTX 6000D.

Reuses the installed vLLM CuTe kernel. Capacity specializations 1..4 are
prepared when weights load; larger batches retain the original GEMM path.
No target/draft weight quantization or activation precision change.
"""
import os
import torch
from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)
_ENABLED = os.environ.get('HOMELAB_MIMO_DENSE_CUTE') == '1'
_SHAPES = {(4096, 2048), (4096, 8192)}  # (N,K), measured TP4 weights
_PREPARED = set()

def prepare_homelab_dense(layer: torch.nn.Module) -> None:
    if not _ENABLED:
        return
    prefix = getattr(layer, 'prefix', '')
    w = layer.weight
    if w.device.type != 'cuda' or w.dtype != torch.bfloat16 or not w.is_contiguous() or tuple(w.shape) not in _SHAPES:
        return
    props = torch.cuda.get_device_properties(w.device)
    if props.name != 'NVIDIA RTX 6000D' or (props.major, props.minor, props.multi_processor_count) != (12, 0, 156):
        return
    from vllm.model_executor.kernels.linear.cute_dsl.ll_bf16 import ll_bf16_gemm_kernel
    n, k = w.shape
    ll_bf16_gemm_kernel.warmup(shapes=[(k, n)], m_values=[1, 2, 3, 4])
    _PREPARED.add((w.device.index, n, k))
    layer._homelab_dense_cute = True
    logger.info('HOMELAB BF16 CuTe projection prepared: %s N=%d K=%d capacity=1..4; other rows use original GEMM', prefix, n, k)

def _impl(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    from vllm.model_executor.kernels.linear.cute_dsl.ll_bf16 import ll_bf16_gemm, ll_bf16_gemm_kernel
    n, k = weight.shape
    if (weight.device.index, n, k) not in _PREPARED:
        raise RuntimeError('Homelab dense projection was not prepared for this capacity')
    # vLLM reuses compiled graphs without shape guards. Keep row dispatch
    # inside this opaque custom op, not in the traced Python model branch.
    if not 1 <= x.shape[0] <= 4:
        return torch.nn.functional.linear(x, weight)
    x = x.contiguous()
    key = ll_bf16_gemm_kernel.dispatch(M=x.shape[0], K=k, N=n)
    if key.backend != 'dotprod' or (key.M, key.K, key.bs) not in ll_bf16_gemm_kernel._compiled_cache:
        raise RuntimeError('Homelab dense projection cannot compile during replay')
    return ll_bf16_gemm(x, weight).to(torch.bfloat16)

def _fake(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return x.new_empty((x.shape[0], weight.shape[0]))

direct_register_custom_op(op_name='homelab_mimo_dense_bf16', op_func=_impl, fake_impl=_fake)

def use_homelab_dense(layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None) -> bool:
    return (getattr(layer, '_homelab_dense_cute', False) and bias is None
            and x.ndim == 2 and x.dtype == torch.bfloat16)
