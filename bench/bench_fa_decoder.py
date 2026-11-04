#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
bench_fa_decoder.py -- the decoder side of cells 2/3: replace MSMN's naive softmax attention
with a memory-efficient path (variant 1c).

MSMN semantics:
    alpha = softmax(score, dim=2)                   over the text axis L
    m     = einsum('abd,aebc->aedc', h, alpha)      (B, C, A, M), v is the full h
    m     = m.max(-1)[0]                            (B, C, A), max-pool over heads

Variant 1c maps this onto FA with q = codes, k = text, v = h broadcast over heads. Because
the q/k head width (128) differs from v's (512) while SDPA requires all three to match, it
zero-pads q/k and splits v:

    k_pad = pad(z_r, 0 -> 128)   (B, L, M, 256)     padding with 0 keeps the dot product
    out_j = sdpa(q_pad, k_pad, v_j, scale=1.0)      MSMN has no 1/sqrt(d) scaling

The cost is computing q.k twice over 256 dims, taking attention from 2.92 to 4.68 TFLOP
(1.60x); what it buys is never materialising the score tensor.
"""
import argparse, torch, torch.nn as nn, torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

ap = argparse.ArgumentParser()
ap.add_argument("--B", type=int, default=8)
ap.add_argument("--L", type=int, default=4000)
ap.add_argument("--C", type=int, default=8921)
ap.add_argument("--A", type=int, default=512)
ap.add_argument("--M", type=int, default=4)
ap.add_argument("--dtype", default="both", choices=["fp32", "bf16", "both"])
ap.add_argument("--mode", default="both", choices=["naive", "fa", "both"])
ap.add_argument("--iters", type=int, default=3)
args = ap.parse_args()

dev = "cuda"
B, L, C, A, M = args.B, args.L, args.C, args.A, args.M
d = A // M
VPAD = 256                     # 1c: zero-pad q/k to 256 and split v into 2x256


# ---------------------------------------------------------------- equivalence check
def check_equivalence(mdtype=torch.float32, b=2, c=64, l=128):
    torch.manual_seed(0)
    z = torch.randn(b, l, M, d, device=dev, dtype=mdtype)
    u = torch.randn(c, M, d, device=dev, dtype=mdtype)
    h = torch.randn(b, l, A, device=dev, dtype=mdtype)

    # naive: materialise the score
    score = torch.einsum('abcd,ecd->aebc', z, u)
    alpha = F.softmax(score, dim=2)
    m_naive = torch.einsum('abd,aebc->aedc', h, alpha).max(-1)[0]

    # variant 1c
    # SDPA's 4-D layout is (B, H, L, E), unlike flash-attn's (B, L, H, E), hence the permute
    q_pad = F.pad(u.unsqueeze(0).expand(b, -1, -1, -1), (0, VPAD - d))   # (b, C, M, 256)
    k_pad = F.pad(z, (0, VPAD - d))                                      # (b, L, M, 256)
    q_ = q_pad.permute(0, 2, 1, 3)                                       # (b, M, C, 256)
    k_ = k_pad.permute(0, 2, 1, 3)                                       # (b, M, L, 256)
    outs = []
    for j in range(A // 256):
        v_j = h[..., j * 256:(j + 1) * 256].unsqueeze(2).expand(-1, -1, M, -1)
        v_j = v_j.permute(0, 2, 1, 3)                                    # (b, M, L, 256)
        outs.append(F.scaled_dot_product_attention(q_, k_, v_j, scale=1.0))
    m_fa = torch.cat(outs, dim=-1).permute(0, 2, 1, 3).max(dim=2)[0]      # (b, C, 512)

    err = (m_fa - m_naive).abs().max().item()
    rel = err / m_naive.abs().max().item()
    print("  [equivalence check fp32] max|d| = %.3e  relative = %.3e  %s"
          % (err, rel, "equivalent elementwise" if rel < 1e-4 else "NOT equivalent"))
    return rel < 1e-4


# ---------------------------------------------------------------- the two implementations
def build(mdtype):
    dt = torch.float32 if mdtype == "fp32" else torch.bfloat16
    W = nn.Linear(A, A).to(dev, dt)
    u_reduce = nn.Linear(A, d).to(dev, dt)
    label_feat = torch.randn(C * M, A, device=dev, dtype=dt)
    h = torch.randn(B, L, A, device=dev, dtype=dt, requires_grad=True)
    return dt, W, u_reduce, label_feat, h


def naive_step(dt, W, u_reduce, label_feat, h):
    z = torch.tanh(W(h))
    z_r = z.reshape(B, L, M, d)
    u_r = u_reduce(label_feat.reshape(C, M, A))
    score = torch.einsum('abcd,ecd->aebc', z_r, u_r)
    alpha = F.softmax(score, dim=2)
    m = torch.einsum('abd,aebc->aedc', h, alpha).max(-1)[0]
    m.sum().backward()


def fa_step(dt, W, u_reduce, label_feat, h, backend=None):
    z = torch.tanh(W(h))
    z_r = z.reshape(B, L, M, d)
    u_r = u_reduce(label_feat.reshape(C, M, A))
    q_pad = F.pad(u_r.unsqueeze(0).expand(B, -1, -1, -1), (0, VPAD - d))
    k_pad = F.pad(z_r, (0, VPAD - d))
    q_ = q_pad.permute(0, 2, 1, 3)      # (B, M, C, 256), SDPA layout
    k_ = k_pad.permute(0, 2, 1, 3)      # (B, M, L, 256)
    ctx = sdpa_kernel(backend) if backend is not None else _nullcontext()
    with ctx:
        outs = []
        for j in range(A // 256):
            v_j = h[..., j * 256:(j + 1) * 256].unsqueeze(2).expand(-1, -1, M, -1)
            v_j = v_j.permute(0, 2, 1, 3)          # (B, M, L, 256)
            outs.append(F.scaled_dot_product_attention(q_, k_, v_j, scale=1.0))
    m = torch.cat(outs, dim=-1).permute(0, 2, 1, 3).max(dim=2)[0]   # (B, C, 512)
    m.sum().backward()


from contextlib import nullcontext as _nullcontext


def timeit(fn, warmup=2, iters=None):
    iters = iters or args.iters
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters / 1e3


print("device %s | torch %s | B=%d L=%d C=%d A=%d M=%d d=%d"
      % (torch.cuda.get_device_name(0), torch.__version__, B, L, C, A, M, d))
check_equivalence()

results = {}
for mdtype in (["fp32", "bf16"] if args.dtype == "both" else [args.dtype]):
    print("\n" + "=" * 72)
    print("dtype = %s   (the score tensor would be %.2f GB if materialised)" % (mdtype, B * C * L * M * 4 / 1e9))
    print("=" * 72)
    for mode in (["naive", "fa"] if args.mode == "both" else [args.mode]):
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        dt, W, u_reduce, label_feat, h = build(mdtype)
        backend = None
        bname = "SDPA automatic"
        if mode == "fa":
            backend = SDPBackend.EFFICIENT_ATTENTION if mdtype == "fp32" else SDPBackend.FLASH_ATTENTION
            bname = backend.name
        try:
            t = timeit(lambda: naive_step(dt, W, u_reduce, label_feat, h) if mode == "naive"
                       else fa_step(dt, W, u_reduce, label_feat, h, backend))
            peak = torch.cuda.max_memory_allocated() / 1e9
            tag = "naive softmax (materialised)" if mode == "naive" else "variant 1c (SDPA)"
            print("  %-28s fwd+bwd : %7.1f ms   peak %6.2f GB   [%s]"
                  % (tag, t * 1e3, peak, bname))
            results[(mdtype, mode)] = (t, peak)
        except torch.OutOfMemoryError:
            print("  %-22s fwd+bwd : ***OOM*** (>23 GB)" %
                  ("naive softmax (materialised)" if mode == "naive" else "variant 1c (SDPA)"))
            results[(mdtype, mode)] = (None, None)
        except Exception as ex:
            print("  %-22s fwd+bwd : ***%s***  %s" %
                  (("naive softmax (materialised)" if mode == "naive" else "variant 1c (SDPA)"),
                   type(ex).__name__, str(ex)[:150]))
            results[(mdtype, mode)] = (None, None)
        del W, u_reduce, label_feat, h; torch.cuda.empty_cache()

print("\n" + "=" * 72)
print("decoder summary (B=%d per card)" % B)
print("=" * 72)
print("  %-8s %22s %22s %10s" % ("dtype", "naive (score materialised)", "variant 1c (SDPA)", "speedup"))
for mdtype in (["fp32", "bf16"] if args.dtype == "both" else [args.dtype]):
    n = results.get((mdtype, "naive"), (None, None))[0]
    f_ = results.get((mdtype, "fa"), (None, None))[0]
    if n and f_:
        print("  %-8s %15.1f ms %15.1f ms %9.2fx" % (mdtype, n * 1e3, f_ * 1e3, n / f_))
    else:
        print("  %-8s %15s %15s %10s"
              % (mdtype, "OOM" if not n else "%.1f ms" % (n * 1e3),
                 "OOM" if not f_ else "%.1f ms" % (f_ * 1e3), "-"))
