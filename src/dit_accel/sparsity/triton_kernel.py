"""Column-sparse 1x1 Conv2d via Triton, with a dense fallback."""
import torch
import torch.nn.functional as F


try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False


if _TRITON_AVAILABLE:

    @triton.jit
    def _sparse_matmul_kernel(
        x_ptr, w_ptr, bias_ptr, out_ptr, chan_mask_ptr,
        M, N, K, HW,
        stride_xm, stride_xk,
        stride_wk, stride_wn,
        stride_om, stride_on,
        stride_maskb, stride_maskk,
        HAS_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        # BLOCK_M must divide HW so a tile stays inside one batch row.
        batch_idx = (pid_m * BLOCK_M) // HW

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
        w_ptrs = w_ptr + (offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn)

        k_tiles = tl.cdiv(K, BLOCK_K)
        for k_tile in range(k_tiles):
            k_start = k_tile * BLOCK_K
            mask_offs = k_start + offs_k
            tile_mask = tl.load(
                chan_mask_ptr + batch_idx * stride_maskb + mask_offs * stride_maskk,
                mask=mask_offs < K, other=0,
            )
            tile_active = tl.max(tile_mask.to(tl.int32))
            if tile_active > 0:
                a = tl.load(
                    x_ptrs,
                    mask=(offs_m[:, None] < M) & ((k_start + offs_k)[None, :] < K),
                    other=0.0,
                )
                b = tl.load(
                    w_ptrs,
                    mask=((k_start + offs_k)[:, None] < K) & (offs_n[None, :] < N),
                    other=0.0,
                )
                acc += tl.dot(a, b)
            x_ptrs += BLOCK_K * stride_xk
            w_ptrs += BLOCK_K * stride_wk

        if HAS_BIAS:
            bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
            acc += bias[None, :]

        out_ptrs = out_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
        out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=out_mask)


def _reduce_to_channel_mask(spatial_mask: torch.Tensor) -> torch.Tensor:
    return spatial_mask.flatten(2).any(dim=-1)


def sparse_conv_point(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    spatial_mask: torch.Tensor | None,
    *,
    force_dense: bool = False,
) -> torch.Tensor:
    """1x1 conv2d that skips K-tiles where every channel is masked out."""
    if force_dense or spatial_mask is None or not _TRITON_AVAILABLE or not x.is_cuda:
        return F.conv2d(x, weight, bias)

    if weight.shape[2] != 1 or weight.shape[3] != 1:
        return F.conv2d(x, weight, bias)

    B, C_in, H, W = x.shape
    C_out = weight.shape[0]
    M = B * H * W
    N = C_out
    K = C_in
    HW = H * W

    BLOCK_M = 64
    BLOCK_N = 128
    BLOCK_K = 64

    if HW % BLOCK_M != 0 or M < BLOCK_M:
        return F.conv2d(x, weight, bias)

    chan_mask = _reduce_to_channel_mask(spatial_mask).contiguous().to(torch.bool)

    x_view = x.permute(0, 2, 3, 1).contiguous().view(M, K)
    w_view = weight.view(C_out, C_in).t().contiguous()

    out_view = torch.empty(M, N, dtype=x.dtype, device=x.device)

    has_bias = bias is not None
    bias_arg = bias if has_bias else out_view

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _sparse_matmul_kernel[grid](
        x_view, w_view, bias_arg,
        out_view, chan_mask,
        M, N, K, HW,
        x_view.stride(0), x_view.stride(1),
        w_view.stride(0), w_view.stride(1),
        out_view.stride(0), out_view.stride(1),
        chan_mask.stride(0), chan_mask.stride(1),
        HAS_BIAS=has_bias,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )

    return out_view.view(B, H, W, C_out).permute(0, 3, 1, 2).contiguous()
