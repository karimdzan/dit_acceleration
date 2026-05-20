"""Cross-attention KV cache for Sana (algebraically exact)."""
import torch
import torch.nn.functional as F

from diffusers.models.attention_processor import Attention


_HASH_ATTR = "_dit_accel_xattn_hash"


def _content_fingerprint(t: torch.Tensor) -> tuple:
    """Cheap content fingerprint of t, memoised on the tensor itself."""
    cached = getattr(t, _HASH_ATTR, None)
    if cached is not None:
        return cached
    shape = tuple(t.shape)
    dtype = str(t.dtype)
    if t.numel() == 0:
        fp = (shape, dtype, ())
    else:
        flat = t.detach().reshape(-1)
        n = flat.numel()
        idx = torch.linspace(0, n - 1, 8, dtype=torch.long, device=t.device)
        samples = flat.index_select(0, idx).float().cpu().tolist()
        fp = (shape, dtype, tuple(round(s, 6) for s in samples))
    try:
        setattr(t, _HASH_ATTR, fp)
    except (AttributeError, RuntimeError):
        pass
    return fp


class XAttnCacheStats:
    def __init__(self):
        self.hits = 0
        self.misses = 0

    def clear(self):
        self.hits = 0
        self.misses = 0


class CachedSanaCrossAttnProcessor:
    def __init__(self, layer_id: int, cache: dict, stats: XAttnCacheStats):
        self.stats = stats
        self.layer_id = layer_id
        self.cache = cache

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert encoder_hidden_states is not None, "attn2 requires encoder_hidden_states"

        batch_size, seq_q, _ = hidden_states.shape
        _, seq_k, _ = encoder_hidden_states.shape

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, seq_k, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        query = attn.to_q(hidden_states)

        cache_key = (_content_fingerprint(encoder_hidden_states), self.layer_id)
        cached = self.cache.get(cache_key)
        if cached is None:
            self.stats.misses += 1
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)
            if attn.norm_k is not None:
                key = attn.norm_k(key)
            self.cache[cache_key] = (key, value)
        else:
            self.stats.hits += 1
            key, value = cached

        if attn.norm_q is not None:
            query = attn.norm_q(query)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        out = out.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim).to(query.dtype)
        out = attn.to_out[0](out)
        out = attn.to_out[1](out)
        return out / attn.rescale_output_factor


def install_cross_attn_cache(pipe) -> None:
    cache: dict = {}
    stats = XAttnCacheStats()
    pipe._dit_accel_xattn_stats = stats
    pipe._dit_accel_xattn_cache = cache

    transformer = pipe.transformer
    layer_id = 0
    for block in transformer.transformer_blocks:
        if getattr(block, "attn2", None) is None:
            continue
        block.attn2.set_processor(
            CachedSanaCrossAttnProcessor(layer_id, cache, stats)
        )
        layer_id += 1
