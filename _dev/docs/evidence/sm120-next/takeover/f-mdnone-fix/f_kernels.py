"""P3 small-op fusion kernels for vLLM 0.31.0 / Qwen3.6-27B-FP8 prefill on SM120.

F1  gdn_norm_quant   : GDN gated RMSNorm (layer_norm_fwd_kernel body, copied) + per-128 FP8 group quant,
                       reading z with its true (strided) layout.  Replaces
                       [z.reshape copy] + layer_norm_fwd_kernel + per_token_group_quant_8bit_register_kernel.
F2  conv_post_fused  : causal_conv1d_fwd (width 4, silu) + _fused_post_conv_kernel (l2norm q/k, split, gating)
                       in one kernel, for a single prefill sequence.  Replaces two kernels and the
                       intermediate BF16 conv output round trip.

Numerics are copied expression-for-expression from the official Triton kernels / the CUDA quant op
(scale = fmaxf(absmax, 1e-10) / 448 with IEEE division; q = clamp(v / scale, -448, 448) -> e4m3 RN).
"""
import torch
import triton
import triton.language as tl

FP8_MAX = tl.constexpr(448.0)
QUANT_EPS = tl.constexpr(1e-10)


# --------------------------------------------------------------------------------------------
# F1: gated RMSNorm + FP8 group quant
# --------------------------------------------------------------------------------------------
@triton.jit(do_not_specialize=["stride_q", "M_tok"])
def _gdn_norm_quant_kernel(
    X, Z, W, Q, S, Y, RSTD,
    stride_x_row,      # = 128 (rows are (token, head))
    stride_q,          # = 128, deliberately NOT specialized: caps the fp8 store vector at 8 bytes so the
                       # coalescer keeps the official [1,8] bf16 layout (needed for bit-identical sum order)
    stride_z_tok,      # row stride between tokens of z (16384 for the qkvz slice)
    stride_z_head,     # = 128
    M_rows,            # tokens * NH
    M_tok,
    eps,
    NH: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ROWS_PER_BLOCK: tl.constexpr,
    SCALE_MODE: tl.constexpr,
    STORE_Y: tl.constexpr,
):
    row_start = tl.program_id(0) * ROWS_PER_BLOCK
    rows = row_start + tl.arange(0, ROWS_PER_BLOCK)
    cols = tl.arange(0, BLOCK_N)
    X_base = X + rows[:, None] * stride_x_row + cols[None, :]
    row_mask = rows[:, None] < M_rows
    col_mask = cols[None, :] < N
    mask = row_mask & col_mask
    x = tl.load(X_base, mask=mask, other=0.0).to(tl.float32)

    # --- identical to layer_norm_fwd_kernel (IS_RMS_NORM, HAS_Z, NORM_BEFORE_GATE, swish) ---
    # The variance reduction must use the SAME register layout / summation order as the official kernel
    # ([1,8] per thread, 16 lanes per row).  A dedicated load (distinct eviction policy so it is not CSE'd)
    # anchors that layout even though the FP8 store below prefers a 16-wide layout.
    x_r = tl.load(X_base, mask=mask, other=0.0, eviction_policy="evict_last").to(tl.float32)
    xbar = tl.where(mask, x_r, 0.0)
    var = tl.sum(xbar * xbar, axis=1) / N
    rstd = tl.rsqrt(var + eps)
    if STORE_Y:
        tl.store(RSTD + rows, rstd, mask=rows < M_rows)
    w = tl.load(W + cols, mask=cols < N, other=0.0).to(tl.float32)
    x_hat = x * rstd[:, None]
    y = x_hat * w[None, :]
    tok = rows // NH
    head = rows % NH
    Z_base = Z + tok[:, None].to(tl.int64) * stride_z_tok + head[:, None] * stride_z_head + cols[None, :]
    z = tl.load(Z_base, mask=mask, other=0.0).to(tl.float32)
    y *= z * tl.sigmoid(z)

    # --- the official op chain stores y as BF16, then quantizes the BF16 values ---
    yb16 = y.to(tl.bfloat16)
    if STORE_Y:
        tl.store(Y + rows[:, None] * N + cols[None, :], yb16, mask=mask)
    yb = yb16.to(tl.float32)
    absmax = tl.max(tl.abs(yb), axis=1)
    absmax = tl.maximum(absmax, QUANT_EPS)
    scale = tl.extra.cuda.libdevice.div_rn(absmax, FP8_MAX)
    qv = tl.extra.cuda.libdevice.div_rn(yb, scale[:, None])
    qv = tl.minimum(tl.maximum(qv, -FP8_MAX), FP8_MAX)
    tl.store(Q + tl.multiple_of(rows[:, None] * stride_q, [1, 8]) + cols[None, :], qv.to(tl.float8e4nv), mask=mask)
    # scales are (M_tok, NH) with strides (1, M_tok)
    tl.store(S + head.to(tl.int64) * M_tok + tok, scale, mask=rows < M_rows)


def _calc_rows_per_block(m_rows: int, device: torch.device) -> int:
    from vllm.third_party.flash_linear_attention.ops.layernorm_guard import calc_rows_per_block
    return calc_rows_per_block(m_rows, device)


def gdn_norm_quant(q_out, s_out, x, z, weight, eps, y_debug=None, rstd_debug=None):
    """x: [M, NH, 128] contiguous BF16; z: [M, NH, 128] with stride(2)==1, stride(1)==128;
    weight: [128]; q_out: [M, NH*128] fp8; s_out: [M, NH] fp32 with strides (1, M)."""
    M, NH, N = x.shape
    rows = M * NH
    rpb = _calc_rows_per_block(rows, x.device)
    grid = (triton.cdiv(rows, rpb),)
    _gdn_norm_quant_kernel[grid](
        x, z, weight, q_out, s_out, y_debug if y_debug is not None else q_out, rstd_debug if rstd_debug is not None else torch.empty(rows if y_debug is not None else 1, device=x.device, dtype=torch.float32),
        N, N, z.stride(0), z.stride(1), rows, M, eps,
        NH=NH, N=N, BLOCK_N=N, ROWS_PER_BLOCK=rpb, SCALE_MODE=0, STORE_Y=y_debug is not None,
        num_warps=min(max(N // 256, 1), 8),
    )


MIN_FUSED_M = 256


def gdn_norm_quant_op(q_out, s_out, x, z, weight, eps):
    """Guarded entry point: fused kernel for the verified shape class, official chain otherwise."""
    M, NH, N = x.shape
    ok = (x.is_cuda and x.dtype == torch.bfloat16 and z.dtype == torch.bfloat16 and N == 128 and M >= MIN_FUSED_M
          and x.is_contiguous() and z.stride(2) == 1 and z.stride(1) == 128
          and q_out.dtype == torch.float8_e4m3fn and q_out.is_contiguous() and s_out.stride() == (1, M)
          and torch.cuda.get_device_capability(x.device) == (12, 0))
    if ok:
        gdn_norm_quant(q_out, s_out, x, z, weight, eps)
        return
    (q, s), _ = gdn_norm_quant_reference(x, z, weight, eps)
    q_out.copy_(q)
    s_out.copy_(s)


def gdn_norm_quant_reference(x, z, weight, eps, activation="swish"):
    """Official chain: layer_norm_fwd (copying z like rmsnorm_fn does) then the CUDA group-quant op."""
    from vllm.third_party.flash_linear_attention.ops.layernorm_guard import rmsnorm_fn
    from vllm.model_executor.layers.quantization.utils.fp8_utils import per_token_group_quant_fp8
    y = rmsnorm_fn(x, weight, None, z=z, eps=eps, group_size=None, norm_before_gate=True, activation=activation)
    return per_token_group_quant_fp8(
        y.flatten(-2), 128, column_major_scales=True, dtype=torch.float8_e4m3fn, use_ue8m0=False), y


# --------------------------------------------------------------------------------------------
# F2: causal conv1d (width 4, silu) + post-conv prep for ONE prefill sequence
# --------------------------------------------------------------------------------------------
@triton.jit(do_not_specialize=["L", "stride_st_slot", "stride_st_dim", "stride_st_tok"])
def _conv_post_fused_kernel(
    x_ptr,            # [L, dim] (row stride stride_x_tok, channels contiguous) BF16 = in_proj output slice
    w_ptr,            # [dim, 4]
    state_ptr,        # conv_state [slots, dim, 3]
    slot_ptr,         # int32 [1] cache slot for the sequence
    has_init_ptr,     # bool [1]
    a_ptr, b_ptr, A_log_ptr, dt_bias_ptr,
    q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr,
    stride_x_tok: tl.int64,
    stride_w_dim,
    stride_w_width,
    stride_st_slot: tl.int64,
    stride_st_dim,
    stride_st_tok,
    stride_a_tok, stride_b_tok,
    L,
    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    L2NORM_EPS: tl.constexpr, SOFTPLUS_THRESHOLD: tl.constexpr,
    BLOCK_T: tl.constexpr, BK: tl.constexpr,
):
    i_tb = tl.program_id(0)
    c = tl.program_id(1)                      # 128-channel chunk: [0,H) q, [H,2H) k, [2H,2H+HV) v
    HK: tl.constexpr = H * K
    offs_t = i_tb * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_t = offs_t < L
    offs_k = tl.arange(0, BK)
    ch = c * K + offs_k                       # channel index in [0, dim)
    mask_2d = mask_t[:, None] & (offs_k < K)[None, :]

    has_init = tl.load(has_init_ptr).to(tl.int1)
    slot = tl.load(slot_ptr).to(tl.int64)

    # ---- causal conv, identical op order to _causal_conv1d_fwd_kernel (acc = 0; acc += x_tap * w_tap) ----
    acc = tl.zeros((BLOCK_T, BK), dtype=tl.float32)
    for j in tl.static_range(4):
        w_col = tl.load(w_ptr + ch * stride_w_dim + j * stride_w_width)
        idx = offs_t + (j - 3)                # token feeding tap j
        xm = mask_t & (idx >= 0)
        xv = tl.load(x_ptr + idx[:, None] * stride_x_tok + ch[None, :], mask=xm[:, None], other=0.0)
        if j < 3:
            # idx < 0 -> come from conv_state[slot, ch, 3 + idx] when the sequence has an initial state
            sm = mask_t & (idx < 0) & has_init
            sv = tl.load(state_ptr + slot * stride_st_slot + ch[None, :] * stride_st_dim
                         + (3 + idx)[:, None] * stride_st_tok, mask=sm[:, None], other=0.0)
            xv = tl.where((idx >= 0)[:, None], xv, sv)
        acc += xv * w_col[None, :]
    acc = acc / (1 + tl.exp(-acc))
    y = acc.to(tl.bfloat16)                   # the official kernel stores the conv output as BF16

    if c < 2 * H:
        yf = y.to(tl.float32)
        sq = tl.sum(yf * yf, axis=1)
        inv = 1.0 / tl.sqrt(sq + L2NORM_EPS)
        yf = yf * inv[:, None]
        out_off = offs_t[:, None] * HK + (c % H) * K + offs_k[None, :]
        if c < H:
            tl.store(q_ptr + out_off, yf.to(q_ptr.dtype.element_ty), mask=mask_2d)
        else:
            tl.store(k_ptr + out_off, yf.to(k_ptr.dtype.element_ty), mask=mask_2d)
    else:
        i_hv = c - 2 * H
        v_off = offs_t[:, None] * (HV * V) + i_hv * V + offs_k[None, :]
        tl.store(v_ptr + v_off, y, mask=mask_2d)
        A_log_val = tl.load(A_log_ptr + i_hv).to(tl.float32)
        dt_bias_val = tl.load(dt_bias_ptr + i_hv).to(tl.float32)
        a_vals = tl.load(a_ptr + offs_t * stride_a_tok + i_hv, mask=mask_t, other=0).to(tl.float32)
        b_vals = tl.load(b_ptr + offs_t * stride_b_tok + i_hv, mask=mask_t, other=0).to(tl.float32)
        xg = a_vals + dt_bias_val
        sp = tl.where(xg > 0, xg + tl.log(1.0 + tl.exp(-xg)), tl.log(1.0 + tl.exp(xg)))
        sp = tl.where(xg <= SOFTPLUS_THRESHOLD, sp, xg)
        g_vals = -tl.exp(A_log_val) * sp
        beta_vals = tl.sigmoid(b_vals)
        tl.store(g_ptr + offs_t * HV + i_hv, g_vals, mask=mask_t)
        tl.store(beta_ptr + offs_t * HV + i_hv, beta_vals, mask=mask_t)


@triton.jit(do_not_specialize=["L", "stride_st_slot", "stride_st_dim", "stride_st_tok"])
def _conv_state_update_kernel(
    x_ptr, state_ptr, slot_ptr,
    stride_x_tok: tl.int64, stride_st_slot: tl.int64, stride_st_dim, stride_st_tok,
    L, dim,
    BLOCK_N: tl.constexpr,
):
    # state_len (3) <= L: conv_state[slot, ch, i] = x[L-3+i, ch]
    ch = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    m = ch < dim
    slot = tl.load(slot_ptr).to(tl.int64)
    for i in tl.static_range(3):
        v = tl.load(x_ptr + (L - 3 + i) * stride_x_tok + ch, mask=m, other=0.0)
        tl.store(state_ptr + slot * stride_st_slot + ch * stride_st_dim + i * stride_st_tok, v, mask=m)


def conv_post_fused(x, conv_weight, conv_state, slot, has_init, a, b, A_log, dt_bias, H, K, V):
    """x: [L, dim] view (row stride arbitrary, channels contiguous).  Returns q,k,v,g,beta like
    fused_post_conv_prep (q,k: [L,H,K], v: [L,HV,V], g,beta: [L,HV] fp32) and updates conv_state."""
    L, dim = x.shape
    HV = A_log.shape[0]
    assert dim == 2 * H * K + HV * V and K == 128 and V == 128
    dtype = x.dtype
    q = torch.empty(L, H, K, dtype=dtype, device=x.device)
    k = torch.empty(L, H, K, dtype=dtype, device=x.device)
    v = torch.empty(L, HV, V, dtype=dtype, device=x.device)
    g = torch.empty(L, HV, dtype=torch.float32, device=x.device)
    beta = torch.empty(L, HV, dtype=torch.float32, device=x.device)
    BLOCK_T = 16
    grid = (triton.cdiv(L, BLOCK_T), 2 * H + HV)
    _conv_post_fused_kernel[grid](
        x, conv_weight, conv_state, slot, has_init, a, b, A_log, dt_bias, q, k, v, g, beta,
        x.stride(0), conv_weight.stride(0), conv_weight.stride(1),
        conv_state.stride(0), conv_state.stride(1), conv_state.stride(2),
        a.stride(0), b.stride(0), L,
        H=H, HV=HV, K=K, V=V, L2NORM_EPS=1e-6, SOFTPLUS_THRESHOLD=20.0,
        BLOCK_T=BLOCK_T, BK=128, num_warps=4, num_stages=2,
    )
    _conv_state_update_kernel[(triton.cdiv(dim, 256),)](
        x, conv_state, slot, x.stride(0), conv_state.stride(0), conv_state.stride(1), conv_state.stride(2),
        L, dim, BLOCK_N=256, num_warps=4)
    return q, k, v, g, beta


_PREWARMED = set()


def prewarm(device, norm=True, conv=True):
    """Compile (JIT) the kernels once at startup so the first real request does not pay Triton compile time."""
    dev = torch.device(device)
    if norm and "norm" not in _PREWARMED:
        _PREWARMED.add("norm")
        M, NH = 512, 48
        x = torch.zeros(M, NH, 128, dtype=torch.bfloat16, device=dev)
        full = torch.zeros(M, 16384, dtype=torch.bfloat16, device=dev)
        z = full[:, 10240:].reshape(M, NH, 128)
        w = torch.ones(128, dtype=torch.float32, device=dev)
        q = torch.empty(M, NH * 128, dtype=torch.float8_e4m3fn, device=dev)
        s = torch.empty(NH, M, dtype=torch.float32, device=dev).permute(1, 0)
        gdn_norm_quant(q, s, x, z, w, 1e-6)
    if conv and "conv" not in _PREWARMED:
        _PREWARMED.add("conv")
        L, H, HV, K, V = 512, 16, 48, 128, 128
        full = torch.zeros(L, 16384, dtype=torch.bfloat16, device=dev)
        cw = torch.zeros(10240, 4, dtype=torch.bfloat16, device=dev)
        cs = torch.zeros(2, 10240, 3, dtype=torch.bfloat16, device=dev)
        a = torch.zeros(L, HV, dtype=torch.bfloat16, device=dev)
        for has in (False, True):
            conv_post_fused(full[:, :10240], cw, cs, torch.zeros(1, dtype=torch.int32, device=dev),
                            torch.tensor([has], device=dev), a, a.clone(), torch.zeros(HV, device=dev),
                            torch.zeros(HV, device=dev), H, K, V)
    torch.cuda.synchronize(dev)
