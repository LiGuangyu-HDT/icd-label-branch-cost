#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
bench_encoders.py -- the two encoders side by side, forward + backward.

  MSMN side     : 2-layer bidirectional LSTM, rnn_dim=256
  FA-MSMN side  : 4-layer Transformer, h=512, 8 heads (SDPA or flash, as configured)

Both are measured at L=4000, B=8 in train mode, and the SDPA backend actually selected is
recorded explicitly.
"""
import argparse, torch, torch.nn as nn

p = argparse.ArgumentParser()
p.add_argument("--B", type=int, default=8)
p.add_argument("--L", type=int, default=4000)
p.add_argument("--layers", type=int, default=4)
p.add_argument("--h", type=int, default=512)
p.add_argument("--heads", type=int, default=8)
p.add_argument("--ffn", type=int, default=2048)
p.add_argument("--dtype", default="both", choices=["fp32", "bf16", "both"])
args = p.parse_args()

dev = "cuda"
B, L = args.B, args.L
print("device %s | torch %s | CUDA %s | B=%d L=%d"
      % (torch.cuda.get_device_name(0), torch.__version__, torch.version.cuda, B, L))
print("allow_tf32(matmul)=%s | cudnn.allow_tf32=%s"
      % (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32))


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


def which_backends():
    from torch.nn.attention import SDPBackend, sdpa_kernel
    got = []
    try:
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            got.append("FLASH")
    except Exception:
        pass
    try:
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            got.append("EFFICIENT")
    except Exception:
        pass
    return got


def run(dtype):
    dt = torch.float32 if dtype == "fp32" else torch.bfloat16
    print("\n" + "=" * 66)
    print("dtype = %s" % dtype)
    print("=" * 66)

    # ---- MSMN side: 2-layer bidirectional BiLSTM, rnn_dim=256 ----
    lstm = nn.LSTM(100, 256, num_layers=2, bidirectional=True,
                   batch_first=True, dropout=0.2).to(dev, dt)
    x = torch.randn(B, L, 100, device=dev, dtype=dt, requires_grad=True)

    def lstm_step():
        y, _ = lstm(x)
        y.sum().backward()

    t_lstm = timeit(lstm_step)
    fl_lstm = 3 * (4 * (100 * 256 + 256 * 256) * 2 * B * L * 2
                   + 4 * (512 * 256 + 256 * 256) * 2 * B * L * 2)
    print("  [MSMN]    BiLSTM(2 layers, bidir, 256) fwd+bwd : %6.1f ms  %5.2f TFLOP/s"
          % (t_lstm * 1e3, fl_lstm / t_lstm / 1e12))

    # ---- FA-MSMN side: 4-layer TransformerEncoder, h=512, 8 heads ----
    torch.cuda.empty_cache()
    enc = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(args.h, args.heads, args.ffn, dropout=0.1,
                                   batch_first=True, norm_first=False),
        args.layers).to(dev, dt)
    x2 = torch.randn(B, L, args.h, device=dev, dtype=dt, requires_grad=True)

    def tf_step():
        y = enc(x2)
        y.sum().backward()

    backend = "MATH (default)"
    if dtype == "bf16":
        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                t_tf = timeit(tf_step)
            backend = "FLASH_ATTENTION"
        except Exception as ex:
            t_tf = timeit(tf_step)
            backend = "fallback (%s)" % type(ex).__name__
    else:
        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel
            with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                t_tf = timeit(tf_step)
            backend = "EFFICIENT_ATTENTION (mem-efficient)"
        except Exception as ex:
            t_tf = timeit(tf_step)
            backend = "fallback (%s)" % type(ex).__name__

    # Rough estimate: self-attention 2*B*H*L*L*d*2 (qk+av) + FFN 2*B*L*h*ffn*2*2 + proj
    fl_attn = 2 * 2 * B * args.heads * L * L * (args.h // args.heads)
    fl_ffn = 2 * 3 * B * L * args.h * args.ffn
    fl_proj = 2 * 4 * B * L * args.h * args.h
    fl_tf = 3 * (fl_attn + fl_ffn + fl_proj) * args.layers
    print("  [FA-MSMN] TransformerEncoder(%d layers, h=%d, %d heads) fwd+bwd : %6.1f ms  %5.2f TFLOP/s"
          % (args.layers, args.h, args.heads, t_tf * 1e3, fl_tf / t_tf / 1e12))
    print("             SDPA backend: %s" % backend)

    torch.cuda.reset_peak_memory_stats()
    lstm_step()
    torch.cuda.synchronize()
    p1 = torch.cuda.max_memory_allocated() / 1e9
    torch.cuda.reset_peak_memory_stats()
    tf_step()
    torch.cuda.synchronize()
    p2 = torch.cuda.max_memory_allocated() / 1e9
    print("  peak memory: BiLSTM %.2f GB | Transformer %.2f GB" % (p1, p2))
    print("  => the FA-MSMN encoder is %.2fx %s than the MSMN encoder"
          % (max(t_tf, t_lstm) / min(t_tf, t_lstm),
             "faster" if t_tf < t_lstm else "slower"))
    return t_lstm, t_tf


res = {}
if args.dtype in ("fp32", "both"):
    res["fp32"] = run("fp32")
if args.dtype in ("bf16", "both"):
    torch.cuda.empty_cache()
    res["bf16"] = run("bf16")

print("\n" + "=" * 66)
print("summary (L=%d, B=%d)" % (L, B))
print("=" * 66)
print("  %-8s %14s %18s %10s" % ("dtype", "BiLSTM(MSMN)", "Transformer(FA-MSMN)", "FA/MSMN"))
for k, (a, b) in res.items():
    print("  %-8s %11.1f ms %15.1f ms %9.2fx" % (k, a * 1e3, b * 1e3, b / a))
print("\nThe decision below looks at the BiLSTM column only:")
for k, (a, _) in res.items():
    v = (">=0.3s -> outcome C" if a >= 0.3 else
         "0.1-0.3s -> outcome A" if a >= 0.1 else "<=0.1s -> outcome B")
    print("  %-8s BiLSTM %.3f s  → %s" % (k, a, v))
