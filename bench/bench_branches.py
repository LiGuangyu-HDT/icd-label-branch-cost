#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
bench_branches.py -- time the three branches separately, check the FLOP model, prototype the
varlen encoder.

The priority of the whole redesign rests on one estimate: the label branch
(35,684 x 30) is 18.22 TFLOP, 5.7x the text branch (B=16, L=4000) at 3.17 TFLOP. This
script measures it rather than assuming it, and answers four questions:

  1. Is the label branch really the largest term?
  2. How much does BiLSTM -> Transformer lose on the label branch?
  3. How much does varlen (removing the padding) recover?
  4. How much padding waste is there on the text branch?

It also writes the VarlenEncoderLayer prototype; that code went on to become part of
combiner.py.

Usage (NVIDIA machine)
    python3 bench_branches.py                  # full scale
    python3 bench_branches.py --small          # reduced batch, for the trend
    python3 bench_branches.py --lab-rows 8000  # fewer rows if the label branch does not fit
    python3 bench_branches.py --no-fa          # skip flash-attn (if not installed)
"""

import argparse
import sys
import traceback

SEP = "=" * 92
RESULTS = []          # (group, name, ms, TFLOP, achieved TFLOP/s, peak GB)


def hr(t):
    print("\n" + SEP); print(t); print(SEP)


def add(group, name, ms, tflop, mem):
    tps = tflop / (ms / 1e3) if ms > 0 else 0.0
    RESULTS.append((group, name, ms, tflop, tps, mem))
    print("   %-34s %9.1f ms   %8.2f TFLOP   %7.1f TFLOP/s   peak %5.2f GB"
          % (name, ms, tflop, tps, mem))


def note(msg):
    print("   ⚠️ " + msg)


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------
def timeit(fn, warmup=2, iters=5):
    import torch
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def measure(fn, warmup=2, iters=5):
    """Return (ms, peak GB). The peak is sampled inside the timing window only."""
    import torch
    fn()                                   # warm-up, so the allocator is primed
    torch.cuda.synchronize()
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    ms = timeit(fn, warmup, iters)
    mem = torch.cuda.max_memory_allocated() / 1024 ** 3
    return ms, mem


# ---------------------------------------------------------------------------
# FLOP convention, identical to gpu_compare_4090_vs_a100.py
# ---------------------------------------------------------------------------
def flops_lstm(n_tok):
    per = 4 * (100 * 256 + 256 * 256) * 2 * 2 + 4 * (512 * 256 + 256 * 256) * 2 * 2
    return per * n_tok


def flops_transformer(n_seq, seqlen, layers, d=512, dff=2048):
    per = 4 * seqlen * seqlen * d + 4 * seqlen * d * dff      # attn + FFN
    return layers * n_seq * per


def flops_decoder(B, C, L, M, d=512):
    return 2 * B * C * L * M * (d // M) + 2 * B * C * d * M * L   # score + agg


# ---------------------------------------------------------------------------
# Prototype: a varlen (padding-free) Transformer layer. If it pays off, this is what goes
# into combiner.py.
# ---------------------------------------------------------------------------
def make_varlen_layer(d=512, nhead=8, dff=2048, dropout=0.1):
    import torch
    import torch.nn as nn
    from flash_attn import flash_attn_varlen_func

    class VarlenEncoderLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.nhead, self.d, self.hd = nhead, d, d // nhead
            self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
            self.qkv = nn.Linear(d, 3 * d)
            self.proj = nn.Linear(d, d)
            self.ffn = nn.Sequential(nn.Linear(d, dff), nn.GELU(), nn.Linear(dff, d))
            self.drop = nn.Dropout(dropout)

        def forward(self, x, cu, max_len):
            total = x.shape[0]
            qkv = self.qkv(self.norm1(x)).reshape(total, 3, self.nhead, self.hd)
            q, k, v = [t.contiguous() for t in qkv.unbind(1)]
            # softmax_scale=1.0, matching MSMN: the score is not divided by sqrt(d)
            a = flash_attn_varlen_func(q, k, v, cu, cu, max_len, max_len, softmax_scale=1.0)
            x = x + self.drop(self.proj(a.reshape(total, self.d)))
            return x + self.drop(self.ffn(self.norm2(x)))

    return VarlenEncoderLayer()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", action="store_true", help="reduced batch, for a quick look at the trend")
    ap.add_argument("--lab-rows", type=int, default=35684,
                    help="number of label sequences (default C*M = 35,684); lower it if memory is short, "
                     "results are scaled accordingly")
    ap.add_argument("--no-fa", action="store_true", help="skip the flash-attn tests")
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

    if not torch.cuda.is_available():
        sys.exit("no CUDA GPU available; this script needs an NVIDIA machine")

    # ---------------- configuration, same as gpu_compare_4090_vs_a100.py ----------------
    B   = 4 if args.small else 16
    L   = 4000
    D   = 512
    DFF = 2048
    LAYERS = 4
    C, M = 8921, 4
    LAB_ROWS = args.lab_rows
    LAB_PAD  = 30
    LAB_AVG  = 4.43           # measured mean words per synonym
    VOCAB = 150853
    scale = LAB_ROWS / (C * M)

    hr("0. environment and configuration")
    p = torch.cuda.get_device_properties(0)
    print("  GPU      : %s (sm_%d%d, %.1f GB)" % (p.name, p.major, p.minor, p.total_memory / 1024**3))
    print("  torch    : %s | CUDA %s" % (torch.__version__, torch.version.cuda))
    try:
        import flash_attn
        print("  flash_attn:", getattr(flash_attn, "__version__", "unknown"))
        HAS_FA = not args.no_fa
    except ImportError:
        print("  flash_attn: not installed (flash-attn tests skipped)")
        HAS_FA = False
    print(f"\n  B={B}  L={L}  d={D}  d_ff={DFF}  layers={LAYERS}")
    print(f"  C={C}  M={M}  labels {LAB_ROWS} x pad{LAB_PAD} (true mean {LAB_AVG})")
    if scale != 1.0:
        note(f"label rows scaled by {scale:.3f}; scale the label-branch numbers accordingly")
    print("\n  Note: to isolate the encoder cost, embeddings use zero-pad(100->512) rather than a")
    print("        real Linear(100,512); that projection adds ~0.1 TFLOP/forward, under 1%.")

    dev = "cuda"
    emb = nn.Embedding(VOCAB, 100).to(dev)

    # =========================================================================
    hr("1. text branch (note encoding)")
    # =========================================================================
    x_tok = torch.randint(0, VOCAB, (B, L), device=dev)
    lens = torch.full((B,), L, dtype=torch.long)

    # --- 1a. BiLSTM (MSMN baseline) ---
    try:
        lstm = nn.LSTM(100, 128, num_layers=2, bidirectional=True,
                       batch_first=True, dropout=0.1).to(dev)

        def run_lstm():
            e = emb(x_tok)
            packed = pack_padded_sequence(e, lens.cpu(), batch_first=True, enforce_sorted=False)
            out, _ = lstm(packed)
            h, _ = pad_packed_sequence(out, batch_first=True)
            h.sum().backward()

        ms, mem = measure(run_lstm)
        add("text", "BiLSTM 2-layer bidir (packed)", ms, 3 * flops_lstm(B * L) / 1e12, mem)
    except Exception as e:
        note("BiLSTM failed: %s: %s" % (type(e).__name__, str(e)[:120]))

    # --- 1b. 4-layer Transformer (padded to 4000) ---
    try:
        enc_doc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(D, 8, DFF, dropout=0.1, batch_first=True),
            LAYERS).to(dev)

        def run_tf_doc():
            e = emb(x_tok)
            enc_doc(F.pad(e, (0, D - e.shape[-1]))).sum().backward()

        ms, mem = measure(run_tf_doc)
        add("text", f"Transformer {LAYERS}L (pad {L})", ms,
            3 * flops_transformer(B, L, LAYERS) / 1e12, mem)
    except Exception as e:
        note("Transformer(text) failed: %s: %s" % (type(e).__name__, str(e)[:160]))

    # =========================================================================
    hr("2. label branch (code synonym descriptions)")
    # =========================================================================
    lab_tok = torch.randint(0, VOCAB, (LAB_ROWS, LAB_PAD), device=dev)
    real_len = max(1, int(round(LAB_AVG)))
    lab_lens = torch.full((LAB_ROWS,), real_len, dtype=torch.long)

    # --- 2a. BiLSTM (MSMN baseline, packed) ---
    try:
        lstm_lab = nn.LSTM(100, 128, num_layers=2, bidirectional=True,
                           batch_first=True, dropout=0.1).to(dev)
        real_tok = int(lab_lens.sum().item())

        def run_lstm_lab():
            e = emb(lab_tok)
            packed = pack_padded_sequence(e, lab_lens.cpu(), batch_first=True, enforce_sorted=False)
            out, _ = lstm_lab(packed)
            h, _ = pad_packed_sequence(out, batch_first=True)
            h.sum().backward()

        ms, mem = measure(run_lstm_lab, iters=3)
        add("label", f"BiLSTM packed ({real_len} real words)", ms, 3 * flops_lstm(real_tok) / 1e12, mem)
    except Exception as e:
        note("BiLSTM(label) failed: %s: %s" % (type(e).__name__, str(e)[:120]))

    # --- 2b. Transformer padded to 30 (the naive port) ---
    try:
        enc_lab = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(D, 8, DFF, dropout=0.1, batch_first=True),
            LAYERS).to(dev)

        def run_tf_lab_pad():
            e = emb(lab_tok)
            enc_lab(F.pad(e, (0, D - e.shape[-1]))).sum().backward()

        ms, mem = measure(run_tf_lab_pad, iters=3)
        add("label", f"Transformer {LAYERS}L (pad {LAB_PAD})", ms,
            3 * flops_transformer(LAB_ROWS, LAB_PAD, LAYERS) / 1e12, mem)
    except Exception as e:
        note("Transformer(label, padded) failed: %s: %s" % (type(e).__name__, str(e)[:160]))

    # --- 2c. Transformer with varlen, no padding ---
    if HAS_FA:
        try:
            cu = torch.zeros(LAB_ROWS + 1, dtype=torch.int32, device=dev)
            cu[1:] = torch.cumsum(lab_lens.to(torch.int32), 0)
            total = int(cu[-1].item())
            x_flat = torch.randn(total, D, device=dev)

            def run_varlen():
                z = x_flat
                for lyr in vlayers:
                    z = lyr(z, cu, real_len)
                z.sum().backward()

            vlayers = nn.ModuleList([make_varlen_layer(D, 8, DFF) for _ in range(LAYERS)]).to(dev)
            ms, mem = measure(run_varlen, iters=3)
            add("label", f"Transformer {LAYERS}L (varlen, no pad)", ms,
                3 * flops_transformer(total / real_len, real_len, LAYERS) / 1e12, mem)
            note("the varlen row excludes the embedding lookup, which is linear in tokens")
        except Exception as e:
            note("varlen failed: %s: %s" % (type(e).__name__, str(e)[:200]))
            print("      -> fall back to bucketing with plain SDPA")

    # =========================================================================
    hr("3. multi-synonym attention (decoder)")
    # =========================================================================
    dec_B = B
    for attempt in range(2):
        try:
            h_txt = torch.randn(dec_B, L, D, device=dev, requires_grad=True)      # v
            z_txt = torch.randn(dec_B, L, M, D // M, device=dev, requires_grad=True)  # k
            u_cod = torch.randn(C, M, D // M, device=dev, requires_grad=True)     # q

            def run_einsum():
                score = torch.einsum('blmd,cmd->bclm', z_txt, u_cod)
                alpha = torch.softmax(score, dim=2)
                m = torch.einsum('ble,bclm->bcem', h_txt, alpha)
                m.max(-1)[0].sum().backward()

            ms, mem = measure(run_einsum, iters=2)
            add("decoder", f"original einsum (score materialised, B={dec_B})", ms,
                3 * flops_decoder(dec_B, C, L, M) / 1e12, mem)
            break
        except torch.cuda.OutOfMemoryError:
            del h_txt, z_txt, u_cod
            torch.cuda.empty_cache()
            if dec_B > 2:
                note(f"the einsum score tensor OOMs at B={dec_B} ({dec_B*C*L*M*4/1e9:.1f} GB in fp32)")
                print(f"      -> retrying at B={dec_B//2}")
                dec_B //= 2
            else:
                note("still OOM at B=2, skipping the decoder"); break
        except Exception as e:
            note("einsum failed: %s: %s" % (type(e).__name__, str(e)[:160])); break

    if HAS_FA:
        try:
            from flash_attn import flash_attn_func
            q = torch.randn(C, M, D // M, device=dev).unsqueeze(0).expand(dec_B, -1, -1, -1).contiguous().requires_grad_(True)
            k = torch.randn(dec_B, L, M, D // M, device=dev, requires_grad=True)
            v = torch.randn(dec_B, L, D, device=dev).unsqueeze(2).expand(-1, -1, M, -1).contiguous().requires_grad_(True)

            def run_fa_dec():
                o = flash_attn_func(q, k, v, softmax_scale=1.0)
                o.permute(0, 1, 3, 2).max(-1)[0].sum().backward()

            ms, mem = measure(run_fa_dec, iters=2)
            add("decoder", f"FA 1a (headdim_v=512, B={dec_B})", ms,
                3 * flops_decoder(dec_B, C, L, M) / 1e12, mem)
        except Exception as e:
            note("FA decoder failed: %s: %s" % (type(e).__name__, str(e)[:160]))
            print("      -> if headdim_v is unsupported, the 1c variant is the fallback")

    # =========================================================================
    hr("4. HBM bandwidth (is it memory bound?)")
    # =========================================================================
    try:
        big = torch.empty(512 * 1024 * 1024, device=dev)
        ms = timeit(lambda: big.mul_(1.0), iters=5)
        print("   2GB mul_ : %.2f ms -> %.0f GB/s" % (ms, 2 * big.numel() * 4 / (ms / 1e3) / 1e9))
        del big
    except Exception as e:
        note("bandwidth test failed: %s: %s" % (type(e).__name__, str(e)[:120]))

    # =========================================================================
    hr("summary")
    # =========================================================================
    print("   %-8s %-36s %10s %10s %12s %9s" %
          ("group", "impl", "ms", "TFLOP", "TFLOP/s", "peak GB"))
    print("   " + "-" * 90)
    for g, n, ms, tf, tps, mem in RESULTS:
        print("   %-8s %-36s %10.1f %10.2f %12.1f %9.2f" % (g, n, ms, tf, tps, mem))

    # ---- key ratios ----
    def get(group, key):
        for g, n, ms, tf, tps, mem in RESULTS:
            if g == group and key in n:
                return ms
        return None

    doc_tf  = get("text", "Transformer")
    lab_pad = get("label", "pad 30")
    lab_var = get("label", "varlen")
    lab_bl  = get("label", "BiLSTM")

    print("\n   -- key ratios --")
    if doc_tf and lab_pad:
        r = lab_pad / doc_tf
        print(f"   label(pad30) / text wall-clock ratio = {r:.2f}x")
        print(f"      the FLOP estimate is 5.7x -- "
              f"{'same order' if r > 2 else 'far below the estimate'}")
    if lab_pad and lab_bl:
        print(f"   label branch BiLSTM -> Transformer regression = {lab_pad/lab_bl:.2f}x")
    if lab_pad and lab_var:
        print(f"   varlen saving over pad30 = {1 - lab_var/lab_pad:.1%}   (the FLOP estimate is 85%)")

    print("""
   How to read these numbers:
     - if the label/text ratio really exceeds 2x, varlen (or bucketing) is the right move
     - if varlen saves less than 50%, bucketing is simpler and achieves the same
     - if the label branch is far lighter than estimated, reprioritise the decoder first""")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n" + SEP); print("aborted with an exception; full traceback:")
        traceback.print_exc(); sys.exit(1)
