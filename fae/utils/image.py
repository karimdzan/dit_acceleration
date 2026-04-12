import math
import torch


def patchify(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Convert BCHW images into B,N,(P*P*C)."""
    b, c, h, w = images.shape
    assert h % patch_size == 0 and w % patch_size == 0
    gh, gw = h // patch_size, w // patch_size
    x = images.reshape(b, c, gh, patch_size, gw, patch_size)
    x = x.permute(0, 2, 4, 3, 5, 1).reshape(b, gh * gw, patch_size * patch_size * c)
    return x


def unpatchify(tokens: torch.Tensor, patch_size: int, image_size: tuple[int, int]) -> torch.Tensor:
    """Convert B,N,(P*P*C) tokens back to BCHW images."""
    b, n, d = tokens.shape
    h, w = image_size
    gh, gw = h // patch_size, w // patch_size
    expected = gh * gw
    if n != expected:
        raise ValueError(f"Expected {expected} patches, got {n}.")
    c = d // (patch_size * patch_size)
    x = tokens.reshape(b, gh, gw, patch_size, patch_size, c)
    x = x.permute(0, 5, 1, 3, 2, 4).reshape(b, c, h, w)
    return x


def infer_prefix_tokens(sequence_length: int) -> int:
    """Infer how many non-patch tokens precede square patch tokens."""
    for prefix in (0, 1, 5, 4, 2, 8):
        n = sequence_length - prefix
        if n > 0 and int(math.isqrt(n)) ** 2 == n:
            return prefix
    raise ValueError(f"Could not infer prefix tokens for sequence length {sequence_length}.")


def infer_hw_from_tokens(num_patches: int) -> tuple[int, int]:
    side = int(math.isqrt(num_patches))

    if side * side != num_patches:
        raise ValueError(f"Patch sequence length {num_patches} is not a square.")
    return side, side
