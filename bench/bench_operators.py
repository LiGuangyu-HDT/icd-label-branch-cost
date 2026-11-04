#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
bench_operators.py -- measure the five key operators on the target GPU. Self-contained:
no cross-machine calibration values are used.

All five are forward + backward:
    1. SGEMM  [C x L] @ [L x h]          the GEMM underlying the aggregation einsum
    2. einsum score  'abcd,ecd->aebc'    multi-synonym attention scoring (decoder.py)
    3. einsum agg    'abd,aebc->aedc'    aggregation (decoder.py), the largest term here
    4. BiLSTM(2 layers, bidirectional, 256) forward + backward, the MSMN encoder
    5. HBM bandwidth

Verified on 2x RTX 4090 24GB (sm_89).
Usage: python bench_operators.py
"""
import torch
import torch.nn as nn

B, L, h, M, C = 4, 4000, 512, 4, 8921     # reduced batch so 24 GB holds it
dev = "cuda"

print("device: %s | %.1f GB | %d SM"
      % (torch.cuda.get_device_name(0),
         torch.cuda.get_device_properties(0).total_memory / 1e9,
         torch.cuda.get_device_properties(0).multi_processor_count))
print("PyTorch: %s | CUDA %s | allow_tf32(matmul)=%s\n"
      % (torch.__version__, torch.version.cuda, torch.backends.cuda.matmul.allow_tf32))


def timeit(fn, warmup=3, iters=10):
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


# 1) large GEMM
a = torch.randn(C, L, device=dev)
b = torch.randn(L, h, device=dev)
t = timeit(lambda: a @ b, iters=5)
print("SGEMM  [%d x %d] @ [%d x %d] : %6.2f ms  %6.1f TFLOP/s"
      % (C, L, L, h, t * 1e3, 2 * C * L * h / t / 1e12))

# 2) score einsum (same shape as the V2 path in decoder.py)
z = torch.randn(B, L, M, h // M, device=dev)
u = torch.randn(C, M, h // M, device=dev)
t = timeit(lambda: torch.einsum('abcd,ecd->aebc', z, u), iters=5)
fl = 2 * B * C * L * M * (h // M)
print("einsum score          : %6.2f ms  %6.1f TFLOP/s  (B=%d)" % (t * 1e3, fl / t / 1e12, B))

# 3) aggregation einsum
hh = torch.randn(B, L, h, device=dev)
al = torch.rand(B, C, L, M, device=dev)
t = timeit(lambda: torch.einsum('abd,aebc->aedc', hh, al), iters=3)
fl = 2 * B * C * h * M * L
print("einsum agg            : %6.2f ms  %6.1f TFLOP/s  (B=%d)" % (t * 1e3, fl / t / 1e12, B))

# 4) BiLSTM encoder
lstm = nn.LSTM(100, 256, num_layers=2, bidirectional=True, batch_first=True,
               dropout=0.2).to(dev)
x = torch.randn(8, L, 100, device=dev, requires_grad=True)


def lstm_step():
    y, _ = lstm(x)
    y.sum().backward()


t = timeit(lstm_step, iters=5)
fl = 3 * (4 * (100 * 256 + 256 * 256) * 2 * 8 * L * 2
          + 4 * (512 * 256 + 256 * 256) * 2 * 8 * L * 2)
print("BiLSTM(2 layers, bidir) fwd+bwd : %6.2f ms  %6.1f TFLOP/s  (B=8)" % (t * 1e3, fl / t / 1e12))

# 5) HBM bandwidth
big = torch.empty(512 * 1024 * 1024, device=dev)   # 2 GB
t = timeit(lambda: big.mul_(1.0), iters=5)
print("HBM bandwidth (2 GB mul_, read+write): %6.1f GB/s" % (2 * big.numel() * 4 / t / 1e9))
