#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
bench_backends.py -- compare SDPA's two memory-efficient backends at the same dtype:
                     EFFICIENT_ATTENTION (xFormers/cutlass) vs FLASH_ATTENTION (Dao).

Question it answers: in going from cell 4 to cell 5 (fp32/mem-efficient to bf16/flash),
both precision and backend change at once. Holding the dtype fixed sizes that confound.

What is measured, matching the real model:
  1. encoder: 4-layer TransformerEncoder (L=4000, h=512, 8 heads), the text branch
  2. decoder: the multi-synonym attention, variant 1c (C=8921, L=4000, M=4, d=128, v=512)
"""
import argparse, torch, torch.nn as nn, torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

ap = argparse.ArgumentParser()
ap.add_argument("--B", type=int, default=8)
ap.add_argument("--L", type=int, default=4000)
ap.add_argument("--C", type=int, default=8921)
ap.add_argument("--A", type=int, default=512)
ap.add_argument("--M", type=int, default=4)
ap.add_argument("--iters", type=int, default=3)
args = ap.parse_args()

dev = "cuda"
B, L, C, A, M = args.B, args.L, args.C, args.A, args.M
d = A // M

print("device %s | torch %s | B=%d L=%d C=%d" %
      (torch.cuda.get_device_name(0), torch.__version__, B, L, C))


def timeit(fn, warmup=2, iters=None):
    iters = iters or args.iters
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters / 1e3


def try_run(tag, fn, backend):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    try:
        with sdpa_kernel(backend):
            t = timeit(fn)
        peak = torch.cuda.max_memory_allocated() / 1e9
        print("    %-34s %8.1f ms   peak %5.2f GB   [%s]"
              % (tag, t * 1e3, peak, backend.name))
        return t, peak
    except Exception as ex:
        print("    %-34s %s: %s" % (tag, type(ex).__name__, str(ex)[:90]))
        return None, None


# ---------------------------------------------------------------- encoder
print("\n" + "=" * 74)
print("1) encoder: 4-layer TransformerEncoder (L=%d, h=%d, 8 heads, ffn=2048)" % (L, A))
print("=" * 74)
for dt, dtname in [(torch.bfloat16, "bf16"), (torch.float32, "fp32")]:
    enc = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(A, 8, 2048, dropout=0.1, batch_first=True), 4).to(dev, dt)
    x = torch.randn(B, L, A, device=dev, dtype=dt, requires_grad=True)

    def step():
        enc(x).sum().backward()

    print("  dtype=%s" % dtname)
    try_run("EFFICIENT_ATTENTION (xFormers)", step, SDPBackend.EFFICIENT_ATTENTION)
    try_run("FLASH_ATTENTION (Dao)", step, SDPBackend.FLASH_ATTENTION)
    del enc, x; torch.cuda.empty_cache()

# ---------------------------------------------------------------- decoder
print("\n" + "=" * 74)
print("2) decoder: multi-synonym attention, variant 1c (C=%d, L=%d, M=%d, d=%d, v=%d)" % (C, L, M, d, A))
print("=" * 74)
for dt, dtname in [(torch.bfloat16, "bf16"), (torch.float32, "fp32")]:
    W = nn.Linear(A, A).to(dev, dt)
    u_reduce = nn.Linear(A, d).to(dev, dt)
    label_feat = torch.randn(C * M, A, device=dev, dtype=dt)
    h = torch.randn(B, L, A, device=dev, dtype=dt, requires_grad=True)

    def step():
        z = torch.tanh(W(h))
        z_r = z.reshape(B, L, M, d)
        u_r = u_reduce(label_feat.reshape(C, M, A))
        q = F.pad(u_r.unsqueeze(0).expand(B, -1, -1, -1), (0, 256 - d)).permute(0, 2, 1, 3)
        k = F.pad(z_r, (0, 256 - d)).permute(0, 2, 1, 3)
        outs = []
        for j in range(A // 256):
            v = h[..., j * 256:(j + 1) * 256].unsqueeze(2).expand(-1, -1, M, -1).permute(0, 2, 1, 3)
            outs.append(F.scaled_dot_product_attention(q, k, v, scale=1.0))
        torch.cat(outs, dim=-1).permute(0, 2, 1, 3).max(dim=2)[0].sum().backward()

    print("  dtype=%s" % dtname)
    try_run("EFFICIENT_ATTENTION (xFormers)", step, SDPBackend.EFFICIENT_ATTENTION)
    try_run("FLASH_ATTENTION (Dao)", step, SDPBackend.FLASH_ATTENTION)
    del W, u_reduce, label_feat, h; torch.cuda.empty_cache()

print("\nNote: FLASH_ATTENTION being unavailable in fp32 is expected -- it has fp16/bf16 kernels only.")
