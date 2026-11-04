#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
bench_step_2x4090.py -- time one real training step from composed operators, giving min/epoch
directly.

Matches cell 1 of the matrix (MSMN / fp32 / standard softmax):
    BiLSTM encoder (2 layers, bidirectional, rnn_dim=256)  ->  h (B, L, 512)
    MultiLabelMultiHeadLAATV2 decoder

Every measurement is forward + backward, as in a real step, reported separately for the
encoder, the decoder einsum chain, and their sum (per-card wall-clock).

DDP semantics: two 4090s each run B=8 (global batch 16), so per-card time is the global
step latency, plus roughly 10 ms of PCIe allreduce.
"""
import argparse, torch, torch.nn as nn, torch.nn.functional as F
from torch import einsum

p = argparse.ArgumentParser()
p.add_argument("--B", type=int, default=8, help="per-card batch (global 16 = 8+8)")
p.add_argument("--L", type=int, default=4000)
p.add_argument("--C", type=int, default=8921)
p.add_argument("--h", type=int, default=512)
p.add_argument("--heads", type=int, default=4)
p.add_argument("--steps", type=int, default=2982, help="steps per epoch at global batch 16")
p.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
p.add_argument("--allreduce-ms", type=float, default=10.0)
args = p.parse_args()

torch.manual_seed(0)
dev = "cuda"
dt = torch.float32 if args.dtype == "fp32" else torch.bfloat16
B, L, C, h, H = args.B, args.L, args.C, args.h, args.heads
d = h // H

print("device %s | dtype=%s | B=%d L=%d C=%d h=%d H=%d | steps/epoch=%d"
      % (torch.cuda.get_device_name(0), args.dtype, B, L, C, h, H, args.steps))
print("torch %s | CUDA %s | allow_tf32=%s"
      % (torch.__version__, torch.version.cuda, torch.backends.cuda.matmul.allow_tf32))


def timeit(fn, warmup=2, iters=5):
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


# ---------------- 1) BiLSTM encoder ----------------
lstm = nn.LSTM(100, 256, num_layers=2, bidirectional=True,
               batch_first=True, dropout=0.2).to(dev, dt)
x = torch.randn(B, L, 100, device=dev, dtype=dt, requires_grad=True)

def lstm_step():
    y, _ = lstm(x)
    y.sum().backward()

t_lstm = timeit(lstm_step)
print("  [1/2] lstm fwd+bwd : %8.1f ms" % (t_lstm * 1e3), flush=True)

# ---------------- 2) decoder einsum chain ----------------
W = nn.Linear(h, h).to(dev, dt)
u_reduce = nn.Linear(h, d).to(dev, dt)
label_feat = torch.randn(C * H, h, device=dev, dtype=dt)   # U.weight: (C*H, h)
hh = torch.randn(B, L, h, device=dev, dtype=dt, requires_grad=True)

def dec_step():
    z = torch.tanh(W(hh))
    z_r = z.reshape(B, L, H, d)
    u_r = u_reduce(label_feat.reshape(C, H, h))
    score = einsum('abcd,ecd->aebc', z_r, u_r)
    alpha = F.softmax(score, dim=2)
    m = einsum('abd,aebc->aedc', hh, alpha)
    m.sum().backward()

try:
    t_dec = timeit(dec_step, warmup=1, iters=3)
    torch.cuda.reset_peak_memory_stats()
    dec_step()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 1e9
except torch.OutOfMemoryError as ex:
    print("  [2/2] dec fwd+bwd  : ***OOM***  alpha tensor %.2f GB, backward needs 2-3 more"
          % (B * C * L * H * 4 / 1e9))
    print("        -> B does not fit on a %.1f GB card: %s"
          % (torch.cuda.get_device_properties(0).total_memory / 1e9,
             str(ex).split(".")[0]))
    print("\n  total per step: infeasible (lower B, use bf16, or checkpoint the decoder)")
    raise SystemExit(1)

# ---------------- 3) summary ----------------
step = t_lstm + t_dec + args.allreduce_ms / 1e3
print()
print("  lstm  fwd+bwd : %8.1f ms" % (t_lstm * 1e3))
print("  dec   fwd+bwd : %8.1f ms   (score tensor = %.2f GB)"
      % (t_dec * 1e3, B * C * L * H * 4 / 1e9))
print("  allreduce     : %8.1f ms   (PCIe, no NVLink)" % args.allreduce_ms)
print("  ---------------------------")
print("  total per step: %8.1f ms" % (step * 1e3))
print("  peak memory   : %8.2f GB  (card %.1f GB)"
      % (peak, torch.cuda.get_device_properties(0).total_memory / 1e9))
print()
print("  => %.1f min/epoch  (%d steps)" % (step * args.steps / 60, args.steps))
