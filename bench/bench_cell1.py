#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
bench_cell1.py -- run cell 1 (MSMN / fp32) for real at B=8 per card (8+8 DDP, global batch 16).

A full step is: BiLSTM encoder -> decoder (three implementations) -> per-code BCE -> backward.

The three decoders answer two different questions:

  naive : plain softmax, materialising the (B,C,L,M) score, i.e. MSMN's original decoder.
          It OOMs at B=8 per card in fp32, which is why cell 1 could not be run as-is.

  ckpt  : the same naive decoder with the code axis split into blocks and
          torch.utils.checkpoint per block. Peak memory drops, FLOPs do not change, and it
          runs. This is the cell-1 configuration used in the paper, and its gradients are
          checked against naive for equivalence.

  fa    : the variant-1c (SDPA mem-efficient) decoder, identical to cell 2's, so that
          cells 1 and 2 differ only in encoder architecture and the 1->2 contrast is clean.

Usage:
    python bench_cell1.py --mode naive          # expected to OOM
    python bench_cell1.py --mode ckpt
    python bench_cell1.py --mode fa
"""
import argparse, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.nn.attention import SDPBackend, sdpa_kernel

ap = argparse.ArgumentParser()
ap.add_argument("--mode", default="fa", choices=["naive", "ckpt", "fa"])
ap.add_argument("--B", type=int, default=8, help="per-card batch (8+8 gives global 16)")
ap.add_argument("--L", type=int, default=4000)
ap.add_argument("--C", type=int, default=8921)
ap.add_argument("--A", type=int, default=512)
ap.add_argument("--M", type=int, default=4)
ap.add_argument("--chunks", type=int, default=4)
ap.add_argument("--steps", type=int, default=2982)
ap.add_argument("--allreduce-ms", type=float, default=10.0)
ap.add_argument("--iters", type=int, default=3)
ap.add_argument("--backend", default="auto", choices=["auto","efficient","flash"])
ap.add_argument("--dtype", default="fp32", choices=["fp32","bf16"])
ap.add_argument("--encoder", default="lstm", choices=["lstm","transformer"])
args = ap.parse_args()

dev = "cuda"
B, L, C, A, M = args.B, args.L, args.C, args.A, args.M
d = A // M
VPAD = 256

print("mode=%s | encoder=%s | dtype=%s | backend=%s | B=%d per card L=%d C=%d A=%d M=%d"
      % (args.mode, args.encoder, args.dtype, args.backend, B, L, C, A, M))
print("torch %s | %s" % (torch.__version__, torch.cuda.get_device_name(0)))


# ------------------------------------------------------------------ model parts
def build():
    dt = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    if args.encoder == "lstm":
        enc = nn.LSTM(100, 256, num_layers=2, bidirectional=True,
                      batch_first=True, dropout=0.2).to(dev, dt)
        x = torch.randn(B, L, 100, device=dev, dtype=dt)
    else:
        enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(A, 8, 2048, dropout=0.1, batch_first=True),
            4).to(dev, dt)
        x = torch.randn(B, L, A, device=dev, dtype=dt)
    W = nn.Linear(A, A).to(dev, dt)
    u_reduce = nn.Linear(A, d).to(dev, dt)
    label_feat = torch.randn(C * M, A, device=dev, dtype=dt)
    return enc, W, u_reduce, label_feat, x


def decoder_chunk_fn(u_r, h, z_r):
    """Forward for one code block: score, softmax, aggregate, max-pool."""
    score = torch.einsum('abcd,ecd->aebc', z_r, u_r)
    alpha = F.softmax(score, dim=2)
    m = torch.einsum('abd,aebc->aedc', h, alpha)
    return m.max(-1)[0]


def step(mode, enc, W, u_reduce, label_feat, x):
    from contextlib import nullcontext
    _be = {"auto": None, "efficient": SDPBackend.EFFICIENT_ATTENTION,
           "flash": SDPBackend.FLASH_ATTENTION}[args.backend]
    ctx = sdpa_kernel(_be) if _be is not None else nullcontext()
    with ctx:                                  # the backend applies to the whole step
        h = enc(x)[0] if args.encoder == "lstm" else enc(x)
        z = torch.tanh(W(h))
        z_r = z.reshape(B, L, M, d)
        u_all = u_reduce(label_feat.reshape(C, M, A))

        if mode == "naive":
            loss = decoder_chunk_fn(u_all, h, z_r).sum()

        elif mode == "ckpt":
            bounds = [i * (C // args.chunks) for i in range(args.chunks)] + [C]
            m_parts = []
            for i in range(args.chunks):
                m_parts.append(checkpoint(decoder_chunk_fn, u_all[bounds[i]:bounds[i + 1]],
                                          h, z_r, use_reentrant=False))
            loss = torch.cat(m_parts, dim=1).sum()

        elif mode == "fa":
            q_pad = F.pad(u_all.unsqueeze(0).expand(B, -1, -1, -1), (0, VPAD - d))
            k_pad = F.pad(z_r, (0, VPAD - d))
            q_ = q_pad.permute(0, 2, 1, 3)
            k_ = k_pad.permute(0, 2, 1, 3)
            outs = []
            for jj in range(A // 256):
                v_j = h[..., jj * 256:(jj + 1) * 256].unsqueeze(2).expand(-1, -1, M, -1)
                outs.append(F.scaled_dot_product_attention(q_, k_, v_j.permute(0, 2, 1, 3), scale=1.0))
            loss = torch.cat(outs, dim=-1).permute(0, 2, 1, 3).max(dim=2)[0].sum()

        loss.backward()


# -------------------------------------------------- ckpt gradient equivalence check
def check_ckpt_grad():
    torch.manual_seed(0)
    b, l, c = 2, 64, 32
    z_r = torch.randn(b, l, M, d, device=dev, requires_grad=True)
    h = torch.randn(b, l, A, device=dev, requires_grad=True)
    u = torch.randn(c, M, d, device=dev, requires_grad=True)

    m1 = decoder_chunk_fn(u, h, z_r)
    m1.sum().backward()
    g = (u.grad.clone(), h.grad.clone(), z_r.grad.clone())
    for t in (u, h, z_r):
        t.grad = None

    parts = [checkpoint(decoder_chunk_fn, u[i:i + c // 4], h, z_r, use_reentrant=False)
             for i in range(0, c, c // 4)]
    torch.cat(parts, dim=1).sum().backward()
    g2 = (u.grad, h.grad, z_r.grad)

    ok = True
    for i, name in enumerate(["u", "h", "z_r"]):
        delta = (g[i] - g2[i]).abs().max().item()
        scale = g[i].abs().max().item() + 1e-12
        rel = delta / scale
        # Blocking changes summation order, so only reassociation error is possible.
        good = rel < 1e-4
        ok &= good
        print("    grad[%s] max|d| = %.2e  relative = %.2e  %s"
              % (name, delta, rel, "ok" if good else "FAIL"))
    print("  [ckpt gradient equivalence] %s" % ("matches naive (reassociation error only)" if ok else "MISMATCH"))
    del m1, parts
    torch.cuda.empty_cache()


if args.mode == "ckpt":
    print("\n--- checkpoint gradient equivalence check ---")
    check_ckpt_grad()

# ------------------------------------------------------------------ timing
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
enc, W, u_reduce, label_feat, x = build()

def run():
    step(args.mode, enc, W, u_reduce, label_feat, x)

try:
    for _ in range(2):
        run()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(args.iters):
        run()
    e.record()
    torch.cuda.synchronize()
    t = s.elapsed_time(e) / args.iters / 1e3
except torch.OutOfMemoryError as ex:
    print("\n  fwd+bwd : ***OOM***  %s" % str(ex).split(".")[0])
    raise SystemExit(1)

peak = torch.cuda.max_memory_allocated() / 1e9
tot = t + args.allreduce_ms / 1e3
print("\n" + "=" * 62)
print("  mode %s | B per card=%d | %.1f ms/step | peak %.2f GB"
      % (args.mode, B, t * 1e3, peak))
print("  + DDP allreduce %.0f ms  ->  %.1f ms/step" % (args.allreduce_ms, tot * 1e3))
print("  => %.1f min/epoch  (%d steps)" % (tot * args.steps / 60, args.steps))
print("=" * 62)
