#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
bench_masked.py -- how a padding mask changes backend selection and timing.

PyTorch SDPA's FLASH_ATTENTION backend accepts attn_mask=None only (any mask raises
"No available kernel"), while both this model's encoder self-attention and its decoder
multi-synonym attention need a padding mask.

So this script asks: once the real padding mask is present,
  - which backend does SDPA select?
  - how much slower is it than the mask-free flash path?
"""
import torch, torch.nn as nn, torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

B, L, C, A, M = 8, 4000, 8921, 512, 4
d, VCHUNK = A // M, 256
dev, dt = "cuda", torch.bfloat16

print("device %s | dtype=bf16 | B=%d L=%d C=%d A=%d M=%d" %
      (torch.cuda.get_device_name(0), B, L, C, A, M))

# padding mask at the real shape: about 20% of positions are padding
word_mask = torch.ones(B, L, device=dev, dtype=torch.long)
word_mask[:, int(L * 0.8):] = 0
key_pad = (word_mask == 0)
print("padding fraction: %.1f%%\n" % (100 * key_pad.float().mean().item()))


def timeit(fn, warmup=2, iters=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters / 1e3


def run(tag, fn, backend=None):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    try:
        if backend is None:
            t = timeit(fn)
            chosen = "auto"
        else:
            with sdpa_kernel(backend):
                t = timeit(fn)
            chosen = backend.name
        print("   %-40s %8.1f ms  peak %5.2f GB  [%s]"
              % (tag, t * 1e3, torch.cuda.max_memory_allocated() / 1e9, chosen))
        return t
    except Exception as ex:
        print("   %-40s FAIL %s: %s" % (tag, type(ex).__name__, str(ex)[:55]))
        return None


# ---------------------------------------------------------------- encoder
print("=" * 84)
print("encoder: 4-layer TransformerEncoder (src_key_padding_mask)")
print("=" * 84)
enc = nn.TransformerEncoder(
    nn.TransformerEncoderLayer(A, 8, 2048, dropout=0.1, batch_first=True), 4).to(dev, dt)
x = torch.randn(B, L, A, device=dev, dtype=dt, requires_grad=True)
pad_mask = key_pad.clone()
pad_mask[pad_mask.all(dim=1), 0] = False


def enc_nomask():
    enc(x).sum().backward()


def enc_masked():
    enc(x, src_key_padding_mask=pad_mask).sum().backward()


run("no mask (previous baseline)", enc_nomask)
run("no mask + force FLASH", enc_nomask, SDPBackend.FLASH_ATTENTION)
run("masked (automatic)", enc_masked)
run("masked + force EFFICIENT", enc_masked, SDPBackend.EFFICIENT_ATTENTION)
run("masked + force FLASH", enc_masked, SDPBackend.FLASH_ATTENTION)
del enc, x; torch.cuda.empty_cache()

# ---------------------------------------------------------------- decoder
print()
print("=" * 84)
print("decoder: multi-synonym attention, variant 1c (q/k zero-padded to 256, v split in 2)")
print("=" * 84)
W = nn.Linear(A, A).to(dev, dt)
u_reduce = nn.Linear(A, d).to(dev, dt)
label_feat = torch.randn(C * M, A, device=dev, dtype=dt)
h_ref = torch.randn(B, L, A, device=dev, dtype=dt)

attn_mask = torch.zeros(B, 1, 1, L, device=dev, dtype=dt).masked_fill(
    key_pad[:, None, None, :], torch.tensor(-1e6, device=dev, dtype=dt))


def dec(mask):
    # Rebuild the graph each time, or backward runs through it twice
    h = h_ref.detach().clone().requires_grad_(True)
    z_r = torch.tanh(W(h)).reshape(B, L, M, d)
    q_r = u_reduce(label_feat.reshape(C, M, A))
    q_pad = F.pad(q_r.unsqueeze(0).expand(B, -1, -1, -1), (0, VCHUNK - d)).permute(0, 2, 1, 3)
    k_pad = F.pad(z_r, (0, VCHUNK - d)).permute(0, 2, 1, 3)
    out = 0
    for j in range(A // VCHUNK):
        v = h[..., j * VCHUNK:(j + 1) * VCHUNK].unsqueeze(2).expand(-1, -1, M, -1).permute(0, 2, 1, 3)
        out = out + F.scaled_dot_product_attention(q_pad, k_pad, v, attn_mask=mask, scale=1.0).sum()
    out.backward()


run("no mask (previous baseline)", lambda: dec(None))
run("no mask + force FLASH", lambda: dec(None), SDPBackend.FLASH_ATTENTION)
run("masked (automatic)", lambda: dec(attn_mask))
run("masked + force EFFICIENT", lambda: dec(attn_mask), SDPBackend.EFFICIENT_ATTENTION)
run("masked + force FLASH", lambda: dec(attn_mask), SDPBackend.FLASH_ATTENTION)
