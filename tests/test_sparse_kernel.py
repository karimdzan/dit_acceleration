"""Correctness tests for the column-sparse Triton kernel."""
import sys

import torch
import torch.nn.functional as F


def _gpu_required():
    if not torch.cuda.is_available():
        print("CUDA not available; skipping kernel tests.")
        sys.exit(0)


def test_dense_mask_matches_conv2d():
    _gpu_required()
    from dit_accel.sparsity.triton_kernel import sparse_conv_point

    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.bfloat16
    B, C_in, H, W = 2, 128, 8, 8
    C_out = 64
    x = torch.randn(B, C_in, H, W, device=device, dtype=dtype)
    w = torch.randn(C_out, C_in, 1, 1, device=device, dtype=dtype) * 0.02
    bias = torch.randn(C_out, device=device, dtype=dtype) * 0.1
    mask = torch.ones(B, C_in, H, W, device=device, dtype=torch.bool)

    out_kernel = sparse_conv_point(x, w, bias, spatial_mask=mask)
    out_dense = F.conv2d(x, w, bias)
    diff = (out_kernel - out_dense).abs().max().item()
    assert diff < 0.05


def test_partial_mask_matches_masked_conv2d():
    _gpu_required()
    from dit_accel.sparsity.triton_kernel import sparse_conv_point

    torch.manual_seed(1)
    device = "cuda"
    dtype = torch.bfloat16
    B, C_in, H, W = 2, 256, 16, 16
    C_out = 128
    x = torch.randn(B, C_in, H, W, device=device, dtype=dtype)
    w = torch.randn(C_out, C_in, 1, 1, device=device, dtype=dtype) * 0.02
    bias = torch.randn(C_out, device=device, dtype=dtype) * 0.1

    chan_mask = torch.rand(B, C_in, device=device) > 0.5
    mask = chan_mask[:, :, None, None].expand(B, C_in, H, W).contiguous()
    x_masked = x * mask

    out_kernel = sparse_conv_point(x_masked, w, bias, spatial_mask=mask)
    out_ref = F.conv2d(x_masked, w, bias)
    diff = (out_kernel - out_ref).abs().max().item()
    assert diff < 0.1


def test_no_bias():
    _gpu_required()
    from dit_accel.sparsity.triton_kernel import sparse_conv_point

    torch.manual_seed(2)
    device = "cuda"
    dtype = torch.bfloat16
    B, C_in, H, W = 1, 64, 8, 8
    C_out = 32
    x = torch.randn(B, C_in, H, W, device=device, dtype=dtype)
    w = torch.randn(C_out, C_in, 1, 1, device=device, dtype=dtype) * 0.02
    mask = torch.ones(B, C_in, H, W, device=device, dtype=torch.bool)

    out_kernel = sparse_conv_point(x, w, bias=None, spatial_mask=mask)
    out_dense = F.conv2d(x, w, bias=None)
    diff = (out_kernel - out_dense).abs().max().item()
    assert diff < 0.05


def test_cpu_fallback_returns_conv2d():
    if torch.cuda.is_available():
        return
    from dit_accel.sparsity.triton_kernel import sparse_conv_point

    x = torch.randn(1, 16, 4, 4)
    w = torch.randn(8, 16, 1, 1) * 0.1
    bias = torch.randn(8) * 0.1
    mask = torch.ones(1, 16, 4, 4, dtype=torch.bool)
    out = sparse_conv_point(x, w, bias, spatial_mask=mask)
    ref = F.conv2d(x, w, bias)
    assert torch.allclose(out, ref)


def main():
    test_dense_mask_matches_conv2d()
    test_partial_mask_matches_masked_conv2d()
    test_no_bias()
    test_cpu_fallback_returns_conv2d()
    print("OK")


if __name__ == "__main__":
    main()
