"""Ragged (variable-length) attention utilities.

PyTorch SDPA's FLASH_ATTENTION backend accepts ``attn_mask=None`` only -- any mask raises
"RuntimeError: No available kernel" -- yet both the encoder self-attention
(``src_key_padding_mask``) and the decoder's multi-synonym attention need one.

``flash_attn``'s variable-length interface sidesteps this: all sequences in the batch are
concatenated into one long buffer and segment boundaries are described by cumulative
lengths ``cu_seqlens``, so attention runs within segments and no padding (hence no mask) is
ever materialised. The cost is that the representation changes from padded ``(B, L, ...)``
to ragged ``(SL, ...)``.

Scaling differs between the two call sites -- the easiest thing to get wrong:
  * encoder self-attention: standard ``1/sqrt(head_dim)`` (flash_attn's default; do not pass
    ``softmax_scale``);
  * decoder multi-synonym attention: no scaling at all (the original MSMN semantics), so
    ``softmax_scale=1.0`` must be passed explicitly.
"""
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint as _ckpt

try:
    from flash_attn import flash_attn_varlen_func
    HAS_FLASH_ATTN = True
except Exception:                                    # pragma: no cover
    flash_attn_varlen_func = None
    HAS_FLASH_ATTN = False


# SDPA backends for the dense path. Note: under bf16 with attn_mask=None PyTorch picks
# the flash kernel, so a non-flash baseline must request EFFICIENT explicitly (which on
# bucketed shapes is also faster, 560 vs 677 ms). The enum is not picklable, so modules
# store the string only and look it up here.
_DENSE_BACKENDS = None


def dense_backend_enum(name):
    """Map a backend name to the SDPBackend enum ('auto' lets PyTorch choose)."""
    global _DENSE_BACKENDS
    if not name or name == 'auto':
        return None
    if _DENSE_BACKENDS is None:
        from torch.nn.attention import SDPBackend
        _DENSE_BACKENDS = {'efficient': SDPBackend.EFFICIENT_ATTENTION,
                           'flash': SDPBackend.FLASH_ATTENTION,
                           'math': SDPBackend.MATH}
    if name not in _DENSE_BACKENDS:
        raise ValueError('unknown dense SDPA backend %r (auto/efficient/flash/math)' % name)
    return _DENSE_BACKENDS[name]


def require_flash_attn():
    if not HAS_FLASH_ATTN:
        raise RuntimeError(
            "the varlen path requires flash-attn; install it with\n"
            "  export CUDA_HOME=/usr/local/cuda-12.8 FLASH_ATTN_CUDA_ARCHS=89 MAX_JOBS=16\n"
            "  pip wheel flash-attn==2.8.3.post1 --no-build-isolation --no-deps -w wheels/")


#: Required dtype on the varlen path. flash_attn ships fp16/bf16 kernels only.
DEFAULT_FLASH_DTYPE = torch.bfloat16


def _align_flash_dtype(*ts, fallback=None):
    """Cast every tensor passed to flash-attn to one half-precision dtype.

    This must be done by hand for two independent reasons:
    1. ``flash_attn`` is a custom op and does not participate in autocast: Linear emits bf16
       while LayerNorm keeps fp32, so q and v can disagree and flash-attn raises
       "query and value must have the same dtype".
    2. accelerate's autocast wraps ``model.forward`` only. Evaluation goes through
       ``model.predict()``, which is outside it and therefore runs in fp32, so the cast
       here is what keeps evaluation from crashing.
    """
    ref = None
    for t in ts:
        if t.dtype in (torch.float16, torch.bfloat16):
            ref = t.dtype
            break
    if ref is None:
        ref = fallback if fallback is not None else DEFAULT_FLASH_DTYPE
    if ref not in (torch.float16, torch.bfloat16):
        raise RuntimeError(
            "the varlen path needs fp16/bf16 (flash-attn has no fp32 kernel) "
            "but the fallback dtype is %s; pass --bf16." % ref)
    return tuple(t.to(ref) if t.dtype != ref else t for t in ts)


# --------------------------------------------------------------------------- #
# Shape helpers
# --------------------------------------------------------------------------- #
def lengths_from_mask(mask):
    """mask (B, L), 1=real / 0=padding  ->  lengths (B,) int32"""
    return mask.sum(dim=1).to(torch.int32)


def lengths_to_cu_seqlens(lengths):
    """lengths (B,)  ->  cu_seqlens (B+1,) int32 as [0, L0, L0+L1, ...]"""
    lengths = lengths.to(torch.int32)
    cu = torch.zeros(lengths.numel() + 1, dtype=torch.int32, device=lengths.device)
    torch.cumsum(lengths, 0, out=cu[1:])
    return cu


def unpad(padded, mask):
    """(B, L, ...) + (B, L)  ->  (SL, ...) keeping real positions only"""
    return padded[mask.bool()]


def segment_position_ids(lengths, device):
    """Within-segment position ids, restarting at 0 for every segment.

    A flat index would be wrong: the second sequence must not start at the first one's
    length.
    """
    return torch.cat([torch.arange(int(n), device=device, dtype=torch.long)
                      for n in lengths.tolist()])


def segment_max(x_flat, cu_seqlens, n_seg):
    """Per-segment max pooling on a ragged representation.

    x_flat (SL, D) -> (n_seg, D), equivalent to ``x.max(dim=1)[0]`` on the padded form once
    padding is excluded.
    """
    D = x_flat.shape[-1]
    seg_id = torch.repeat_interleave(
        torch.arange(n_seg, device=x_flat.device),
        (cu_seqlens[1:] - cu_seqlens[:-1]).long())
    out = torch.full((n_seg, D), float("-inf"), device=x_flat.device, dtype=x_flat.dtype)
    out.scatter_reduce_(0, seg_id.unsqueeze(-1).expand_as(x_flat), x_flat, reduce="amax",
                        include_self=True)
    # A zero-length segment receives no tokens, so its row would stay at -inf and turn
    # into NaN. Match the padded path, where zero_pad_output makes the max 0. Real data
    # does contain such segments (codes present in the label table with no description).
    empty = (cu_seqlens[1:] - cu_seqlens[:-1]) == 0
    if bool(empty.any()):
        out = out.masked_fill(empty.unsqueeze(-1), 0.0)
    return out


def segment_mean(x_flat, cu_seqlens, n_seg):
    D = x_flat.shape[-1]
    seg_id = torch.repeat_interleave(
        torch.arange(n_seg, device=x_flat.device),
        (cu_seqlens[1:] - cu_seqlens[:-1]).long())
    out = torch.zeros((n_seg, D), device=x_flat.device, dtype=x_flat.dtype)
    out.index_add_(0, seg_id, x_flat)
    cnt = (cu_seqlens[1:] - cu_seqlens[:-1]).to(x_flat.dtype).unsqueeze(-1).clamp(min=1)
    return out / cnt


# --------------------------------------------------------------------------- #
# Within-segment positional encoding (ragged)
# --------------------------------------------------------------------------- #
class VarlenPositionEmbedding(nn.Module):
    """Within-segment positional encoding: ``none`` or ``learn``, as in ``model/pos.py``.

    ``axial`` is Reformer-specific and not supported here.
    """

    def __init__(self, pos_config):
        super().__init__()
        self.mode = pos_config.get("pos_embed", "none")
        self.dim = pos_config["rnn_dim"]
        if self.mode == "none":
            pass
        elif self.mode == "learn":
            self.pos = nn.Embedding(4096, self.dim)      # same bound as pos.py
        else:
            raise NotImplementedError(
                "varlen pos_embed supports none/learn only, got %r" % self.mode)

    def forward(self, x_flat, lengths):
        if self.mode == "none":
            return torch.zeros_like(x_flat)
        ids = segment_position_ids(lengths, x_flat.device)
        if ids.numel() == 0:
            return torch.zeros_like(x_flat)
        return self.pos(ids)


# --------------------------------------------------------------------------- #
# Encoder layer
# --------------------------------------------------------------------------- #
class VarlenEncoderLayer(nn.Module):
    """Structurally equivalent to ``nn.TransformerEncoderLayer`` (post-LN or pre-LN) with
    the self-attention replaced by ``flash_attn_varlen_func``, so no padding mask is needed.

    Scaling follows flash-attn's default ``1/sqrt(head_dim)`` (``softmax_scale`` is not
    passed), matching ``nn.MultiheadAttention``.
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="gelu", norm_first=False, dense_backend=None):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        # Store the string only: an enum would break torch.save(model)
        self.dense_backend = dense_backend or 'efficient'
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.norm_first = norm_first

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.activation = nn.GELU() if activation == "gelu" else nn.ReLU()

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def self_attention(self, x, cu_seqlens, max_len):
        require_flash_attn()
        T = x.shape[0]
        q = self.q_proj(x).view(T, self.nhead, self.head_dim)
        k = self.k_proj(x).view(T, self.nhead, self.head_dim)
        v = self.v_proj(x).view(T, self.nhead, self.head_dim)
        if T == 0:
            return torch.zeros_like(x)
        q, k, v = _align_flash_dtype(q, k, v)      # flash-attn bypasses autocast
        a = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens,
                                   max_len, max_len, dropout_p=0.0, causal=False)
        return self.out_proj(a.reshape(T, self.d_model))

    def _ff(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))

    # Dense (SDPA) attention, same weights and scaling as the flash version. Needed
    # because bucketed shapes share a length (so no mask is required) and because
    # flash-attn has no fp32 kernel. Bit-identical to nn.TransformerEncoder in fp32,
    # 2.8e-03 relative in bf16 -- the noise of swapping kernels at all.
    def dense_self_attention(self, x, attn_mask=None):
        """x: (B, L, H); attn_mask: bool (B,1,1,L) with True = attended, or None."""
        B, L, _ = x.shape
        nh, hd = self.nhead, self.head_dim
        q = self.q_proj(x).view(B, L, nh, hd).transpose(1, 2)
        k = self.k_proj(x).view(B, L, nh, hd).transpose(1, 2)
        v = self.v_proj(x).view(B, L, nh, hd).transpose(1, 2)
        _be = dense_backend_enum(self.dense_backend)
        if _be is None:
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        else:
            from torch.nn.attention import sdpa_kernel
            with sdpa_kernel(_be):
                a = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return self.out_proj(a.transpose(1, 2).reshape(B, L, self.d_model))

    def forward_dense(self, x, attn_mask=None):
        if self.norm_first:
            x = x + self.dropout1(self.dense_self_attention(self.norm1(x), attn_mask))
            x = x + self.dropout2(self._ff(self.norm2(x)))
        else:
            x = self.norm1(x + self.dropout1(self.dense_self_attention(x, attn_mask)))
            x = self.norm2(x + self.dropout2(self._ff(x)))
        return x

    def forward(self, x, cu_seqlens, max_len):
        if self.norm_first:
            x = x + self.dropout1(self.self_attention(self.norm1(x), cu_seqlens, max_len))
            x = x + self.dropout2(self._ff(self.norm2(x)))
        else:
            x = self.norm1(x + self.dropout1(self.self_attention(x, cu_seqlens, max_len)))
            x = self.norm2(x + self.dropout2(self._ff(x)))
        return x


class VarlenEncoderStack(nn.Module):
    def __init__(self, num_layers, d_model, nhead, dim_feedforward, dropout,
                 activation="gelu", norm_first=False, dense_backend=None):
        super().__init__()
        self.layers = nn.ModuleList([
            VarlenEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation,
                               norm_first, dense_backend)
            for _ in range(num_layers)])

    def forward(self, x, cu_seqlens, max_len):
        for layer in self.layers:
            x = layer(x, cu_seqlens, max_len)
        return x

    # Four forwards, mathematically equivalent, differing only in memory/time/precision:
    # unchunked (simplest), chunked at segment boundaries (flash), dense with a padding
    # mask (text branch), bucketed by exact length. At full scale the unchunked forward
    # is 18.48 GB and its backward OOMs; chunked@16384 is 3.19 GB / 469 ms, bucketed
    # 3.30 GB / 677 ms, padded 8.27 GB / 1883 ms.
    def forward_chunked(self, x, cu_seqlens, chunk_tokens, use_ckpt=True):
        """Chunk at segment boundaries: attention is already per-segment, so nothing crosses
        a boundary and the result is exact."""
        ys = []
        for xc, cuc, ml in segment_chunks(x, cu_seqlens, chunk_tokens):
            def body(t, cuc=cuc, ml=ml):
                for layer in self.layers:
                    t = layer(t, cuc, ml)
                return t
            ys.append(_ckpt(body, xc, use_reentrant=False) if use_ckpt else body(xc))
        return torch.cat(ys, 0)

    def forward_dense_pad(self, x, word_mask, chunk_tokens=0, use_ckpt=True):
        """Dense with a padding mask; the text branch takes this path (lengths within a batch
        are similar, so little is wasted).

        ``word_mask`` is the opposite of nn.MultiheadAttention's key_padding_mask:
        here True means attended (SDPA's bool-mask convention).
        """
        attn_mask = word_mask.bool()[:, None, None, :] if word_mask is not None else None

        def body(t, am):
            for layer in self.layers:
                t = layer.forward_dense(t, am)
            return t

        if chunk_tokens and x.shape[1] and x.shape[0] * x.shape[1] > chunk_tokens:
            rows = max(1, chunk_tokens // x.shape[1])
            outs = []
            for i in range(0, x.shape[0], rows):
                xc = x[i:i + rows].contiguous()
                am = attn_mask[i:i + rows] if attn_mask is not None else None
                outs.append(_ckpt(body, xc, am, use_reentrant=False) if use_ckpt
                            else body(xc, am))
            out = torch.cat(outs, 0)
        else:
            out = body(x, attn_mask)
        if word_mask is not None:
            out = out * word_mask.unsqueeze(-1).to(out.dtype)
        return out

    def forward_bucketed(self, x, lengths, chunk_tokens, use_ckpt=True):
        """Bucket by exact length: uniform within a bucket, so no padding and no mask.

        ``x`` is ragged (SL, H) and the output is (SL, H) in the original segment order.
        Zero-length segments enter no bucket and keep an output of 0, matching the padded
        path's ``zero_pad_output`` (LabelEncoder handles this).
        """
        starts = segment_starts(lengths, x.device)
        # Do not assume the output dtype equals x's: under autocast, LayerNorm and the
        # residual stay fp32, so the first layer already returns fp32.
        out = None
        for L, idx in bucket_batches(lengths, chunk_tokens):
            gidx = starts[idx].unsqueeze(1) + torch.arange(L, device=x.device)
            xb = x[gidx]                                  # (n, L, H) real tokens only

            def body(t):
                for layer in self.layers:
                    t = layer.forward_dense(t, None)
                return t

            xb = _ckpt(body, xb, use_reentrant=False) if use_ckpt else body(xb)
            if out is None:
                out = x.new_zeros(x.shape, dtype=xb.dtype)
            out.index_put_((gidx.reshape(-1),), xb.reshape(-1, xb.shape[-1]))
        if out is None:                                   # all segments empty
            out = torch.zeros_like(x)
        return out


# --------------------------------------------------------------------------- #
# Indexing helpers for chunking and bucketing
# --------------------------------------------------------------------------- #
def segment_starts(lengths, device=None):
    """Segment lengths (N,)  ->  start offsets (N,) in the ragged buffer."""
    lengths = lengths.to(device) if device is not None else lengths
    z = torch.zeros(1, dtype=torch.long, device=lengths.device)
    return torch.cat([z, torch.cumsum(lengths, 0)[:-1]]).long()


def bucket_batches(lengths, chunk_tokens):
    """Bucket by exact length  ->  [(L, idx), ...], each sub-batch <= chunk_tokens.

    Zero-length segments are skipped (an empty bucket breaks attention). Real data yields
    30 buckets for lengths 1..30; the largest is L=4 with 6,435 segments.
    """
    by_len = {}
    for i, L in enumerate(lengths.tolist()):
        if L > 0:
            by_len.setdefault(L, []).append(i)
    batches = []
    for L in sorted(by_len):
        idxs = by_len[L]
        per = max(1, chunk_tokens // L) if chunk_tokens else len(idxs)
        for s in range(0, len(idxs), per):
            batches.append((L, torch.tensor(idxs[s:s + per], dtype=torch.long)))
    return batches


def segment_chunks(x, cu_seqlens, chunk_tokens):
    """Split ragged x at segment boundaries into <=chunk_tokens blocks."""
    lens = (cu_seqlens[1:] - cu_seqlens[:-1])
    groups, cur, tok = [], [], 0
    n = int(lens.numel())
    for i in range(n):
        li = int(lens[i])
        if cur and tok + li > chunk_tokens:
            groups.append(cur)
            cur, tok = [], 0
        cur.append(i)
        tok += li
    if cur:
        groups.append(cur)
    out = []
    for g in groups:
        i0, i1 = g[0], g[-1] + 1
        base = int(cu_seqlens[i0].item())
        xc = x[base:int(cu_seqlens[i1].item())].contiguous()
        cuc = (cu_seqlens[i0:i1 + 1] - base).contiguous()
        ml = int(lens[i0:i1].max().item())
        out.append((xc, cuc, ml))
    return out
