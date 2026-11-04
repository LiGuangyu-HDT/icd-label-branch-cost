#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""bench_enc_ckpt.py -- measure the cost and benefit of gradient checkpointing the encoder.

What is measured
----------------
The Transformer encoder stack only (after the embeddings, before pooling). Checkpointing
helps only where recomputation is cheaper than storage, and for the label branch that is
this segment.

The input is the real label-description feature matrix, matching run_matrix_real.sh:
    --version mimic3 --term_count 4 --sort_method random
    -> 35,684 descriptions x 30 words = 1,070,520 token slots
    -> of which 200,353 are real tokens (5.6 words per description on average)

Two representations, matching the cells:
    padded  (N, L, H)  -> nn.TransformerEncoder + src_key_padding_mask   (cells 2/3/4)
    varlen  (SL, H)    -> VarlenEncoderStack + flash_attn_varlen_func    (cell 5)

Six configurations
------------------
    none      no checkpointing (baseline), which shows where the memory wall is
    layer     one checkpoint per layer
    layer2    only the first two layers checkpointed
    sublayer  attention and FFN blocks checkpointed separately (varlen only)
    chunk     split by token into chunks, one checkpoint per chunk
    chunkL    chunks, plus one checkpoint per layer within each chunk

Why chunking is exact, not approximate
--------------------------------------
varlen: chunks are cut at segment boundaries and attention is already per-segment, so no
information crosses a boundary. padded: chunks are cut along the batch axis and samples are
independent. Both are therefore exact.

Four measurements per configuration
-----------------------------------
forward peak / forward+backward peak / forward+backward wall-clock / max relative gradient
error against the unchecked run.

On dropout
----------
torch.utils.checkpoint(use_reentrant=False) defaults to preserve_rng_state=True, so the
recomputation reuses the same RNG state and dropout is not resampled; results should be
bit-identical. The script measures the relative error rather than assuming it.

Usage
-----
    export ICD_DATA_DIR=/path/to/mimic3_caml
    export ICD_WORD_EMBEDDING=/path/to/word2vec_sg0_100.model
    python -u bench/bench_enc_ckpt.py --mode both --dtype bf16
"""
import argparse
import gc
import os
import sys
import time
import traceback

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as ckpt_fn

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("ICD_REPO", os.path.join(HERE, "..", "model", "main"))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

WE_DEFAULT = os.environ.get("ICD_WORD_EMBEDDING", "")   # required: word2vec model path


# --------------------------------------------------------------------------- #
# arguments
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="both", choices=["padded", "varlen", "both"])
    p.add_argument("--configs", default="none,layer,layer2,sublayer,chunk,chunkL")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"],
                   help="bf16 = accelerate mixed_precision='bf16' (fp32 parameters with autocast)")
    p.add_argument("--chunk_tokens", type=int, default=32768,
                   help="max real tokens per chunk (converted to a row count for padded)")
    # Same as run_matrix_real.sh
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--h", type=int, default=512)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--ffn", type=int, default=2048)
    p.add_argument("--dp", type=float, default=0.1)
    p.add_argument("--norm_first", action="store_true")
    p.add_argument("--label_truncate_length", type=int, default=30)
    p.add_argument("--term_count", type=int, default=4)
    p.add_argument("--sort_method", default="random")
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--iters", type=int, default=3)
    p.add_argument("--dev", default="cuda:0")
    p.add_argument("--n_seg", type=int, default=35684,
                   help="number of descriptions under --dry_data")
    p.add_argument("--seg_limit", type=int, default=0,
                   help="use only the first N label descriptions (0 = all), so the "
                        "checkpointing overhead and numerical equivalence can be measured at a "
                        "scale the unchecked configuration fits")
    p.add_argument("--dry_data", action="store_true",
                   help="skip real data and use synthetic label features (a smoke test of the script itself)")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Real label features, without loading the 405 MB mimic3_train.json
# --------------------------------------------------------------------------- #
def real_label_features(args):
    """Return (c_input_word, c_word_mask, c_word_sent), matching MimicFullDataset(mode='train').

    MimicFullDataset.__init__ would load the 405 MB train json, which this script does not
    need, so only the code-table, synonym, tokenisation, vocabulary and padding steps are
    reproduced here.
    """
    from data_util import (MimicFullDataset, create_main_code, load_full_codes,
                           load_vocab)
    from constant import MIMIC_3_DIR
    from nltk.tokenize import RegexpTokenizer

    class LabelOnly(MimicFullDataset):
        def __init__(self, version, vocab_path, truncate_length,
                     label_truncate_length, term_count, sort_method):
            self.version = version
            self.mode = "train"
            self.path = os.path.join(MIMIC_3_DIR, "%s_train.json" % version)
            self.train_path = os.path.join(MIMIC_3_DIR, "train_full.csv")
            self.vocab_path = vocab_path
            self.word2id, self.id2word = load_vocab(vocab_path)
            self.truncate_length = truncate_length
            self.ind2c, _ = load_full_codes(self.train_path, version=version)
            self.c2ind = {c: i for i, c in self.ind2c.items()}
            self.code_count = len(self.ind2c)
            self.ind2mc, self.mc2ind = create_main_code(self.ind2c)
            self.main_code_count = len(self.ind2mc)
            self.tokenizer = RegexpTokenizer(r"\w+")
            self.len = 0
            self.label_truncate_length = label_truncate_length
            self.term_count = term_count
            self.sort_method = sort_method
            self.prepare_label_feature(label_truncate_length)

    ds = LabelOnly(version="mimic3", vocab_path=args.we, truncate_length=4000,
                   label_truncate_length=args.label_truncate_length,
                   term_count=args.term_count, sort_method=args.sort_method)
    return ds.c_input_word, ds.c_word_mask, ds.code_count


def synthetic_label_features(args, n_seg=None):
    """Synthetic data with the real segment-length distribution (uniform 1..30)."""
    if n_seg is None:
        n_seg = args.n_seg
    torch.manual_seed(0)
    lengths = torch.randint(1, args.label_truncate_length + 1, (n_seg,))
    mask = torch.zeros(n_seg, args.label_truncate_length, dtype=torch.long)
    for i, L in enumerate(lengths.tolist()):
        mask[i, :L] = 1
    words = torch.randint(2, 1000, (n_seg, args.label_truncate_length))
    return words, mask, 8921


# --------------------------------------------------------------------------- #
# Stack construction: fixed seed, so every configuration uses identical weights
# --------------------------------------------------------------------------- #
def _fp32_varlen_note(args):
    print("   [warn] flash-attn ships fp16/bf16 kernels only, so fp32 + varlen is not a legal\n"
          "          configuration: every varlen/fp32 row below fails in out_proj with\n"
          "          'mat1 and mat2 must have the same dtype'. This is by design, not a bug.")


def build_stack(kind, args, seed=0):
    torch.manual_seed(seed)
    if kind == "varlen":
        from model.varlen import VarlenEncoderLayer
        return nn.ModuleList([
            VarlenEncoderLayer(args.h, args.heads, args.ffn, args.dp, "gelu",
                               args.norm_first)
            for _ in range(args.layers)])
    return nn.ModuleList([
        nn.TransformerEncoderLayer(d_model=args.h, nhead=args.heads,
                                   dim_feedforward=args.ffn, dropout=args.dp,
                                   activation="gelu", batch_first=True,
                                   norm_first=args.norm_first)
        for _ in range(args.layers)])


def _ck(fn, *a, **kw):
    return ckpt_fn(fn, *a, use_reentrant=False, **kw)


# ------------------------------- varlen forward ------------------------------- #
def varlen_layer_sub(layer, x, cu, max_len):
    """Sub-layer granularity: attention and FFN blocks checkpointed separately."""
    if layer.norm_first:
        a = _ck(lambda t: layer.self_attention(layer.norm1(t), cu, max_len), x)
        x = x + layer.dropout1(a)
        f = _ck(lambda t: layer._ff(layer.norm2(t)), x)
        return x + layer.dropout2(f)
    x = _ck(lambda t: layer.norm1(
        t + layer.dropout1(layer.self_attention(t, cu, max_len))), x)
    return _ck(lambda t: layer.norm2(t + layer.dropout2(layer._ff(t))), x)


def varlen_chunks(x, cu, chunk_tokens):
    """Split at segment boundaries -> [(xc, cu_c, max_len_c), ...]; exact."""
    lens = (cu[1:] - cu[:-1]).tolist()
    groups, cur, cur_tok = [], [], 0
    for i, li in enumerate(lens):
        if cur and cur_tok + li > chunk_tokens:
            groups.append(cur)
            cur, cur_tok = [], 0
        cur.append(i)
        cur_tok += li
    if cur:
        groups.append(cur)
    out = []
    for g in groups:
        i0, i1 = g[0], g[-1] + 1
        base = int(cu[i0].item())
        xc = x[base:int(cu[i1].item())].contiguous()
        cuc = (cu[i0:i1 + 1] - base).contiguous()
        out.append((xc, cuc, int(max(lens[i0:i1]))))
    return out


def varlen_forward(mods, x, cu, max_len, config, chunk_tokens):
    if config in ("chunk", "chunkL"):
        use_ck = (config == "chunkL")
        ys = []
        for xc, cuc, ml in varlen_chunks(x, cu, chunk_tokens):
            def body(t, cuc=cuc, ml=ml):
                for m in mods:
                    t = _ck(m, t, cuc, ml) if use_ck else m(t, cuc, ml)
                return t
            ys.append(_ck(body, xc))
        return torch.cat(ys, 0)
    for i, m in enumerate(mods):
        if config == "layer" or (config == "layer2" and i < 2):
            x = _ck(m, x, cu, max_len)
        elif config == "sublayer":
            x = varlen_layer_sub(m, x, cu, max_len)
        else:
            x = m(x, cu, max_len)
    return x


# ------------------------------- padded forward ------------------------------- #
def padded_forward(mods, x, mask, config, chunk_rows):
    if config in ("chunk", "chunkL"):
        use_ck = (config == "chunkL")
        ys = []
        for i0 in range(0, x.shape[0], chunk_rows):
            xc = x[i0:i0 + chunk_rows].contiguous()
            mc = mask[i0:i0 + chunk_rows] if mask is not None else None

            def body(t, mc=mc):
                for m in mods:
                    t = _ck(m, t, src_key_padding_mask=mc) if use_ck else \
                        m(t, src_key_padding_mask=mc)
                return t
            ys.append(_ck(body, xc))
        return torch.cat(ys, 0)
    for i, m in enumerate(mods):
        if config == "layer" or (config == "layer2" and i < 2):
            x = _ck(m, x, src_key_padding_mask=mask)
        else:
            x = m(x, src_key_padding_mask=mask)
    return x


# --------------------------------------------------------------------------- #
# Measurement
# --------------------------------------------------------------------------- #
def fmt_gb(nbytes):
    return nbytes / 1024.0 ** 3


def measure(mode, config, mods, x, cu, mask, args):
    """Return a dict; OOM returns {'oom': True, 'peak': ...}."""
    res = {"mode": mode, "config": config, "oom": False}
    if mode == "varlen":
        max_len = int((cu[1:] - cu[:-1]).max().item())
        chunk_rows = None

        def fwd():
            return varlen_forward(mods, x, cu, max_len, config, args.chunk_tokens)

        n_chunks = len(varlen_chunks(x, cu, args.chunk_tokens)) if config.startswith("chunk") else 1
    else:
        L = x.shape[1]
        chunk_rows = max(1, args.chunk_tokens // L)
        n_chunks = (x.shape[0] + chunk_rows - 1) // chunk_rows if config.startswith("chunk") else 1

        def fwd():
            return padded_forward(mods, x, mask, config, chunk_rows)
    res["n_chunks"] = n_chunks

    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if args.dtype == "bf16" \
        else torch.autocast("cuda", enabled=False)

    # ---- 1) forward-only peak ----
    try:
        with torch.no_grad(), ctx:
            fwd()
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        with ctx:
            out = fwd()
        torch.cuda.synchronize()
        res["fwd_peak"] = fmt_gb(torch.cuda.max_memory_allocated())
        del out
        gc.collect()
        torch.cuda.empty_cache()
    except torch.cuda.OutOfMemoryError:
        res["oom"] = True
        res["fwd_peak"] = fmt_gb(torch.cuda.max_memory_allocated())
        gc.collect()
        torch.cuda.empty_cache()
        return res

    # ---- 2) forward+backward peak and time ----
    def step():
        for p in mods.parameters():
            p.grad = None
        with ctx:
            out = fwd()
            loss = out.float().sum()
        loss.backward()

    try:
        step()
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        for _ in range(args.warmup):
            step()
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(args.iters):
            step()
        e.record()
        torch.cuda.synchronize()
        res["bwd_peak"] = fmt_gb(torch.cuda.max_memory_allocated())
        res["reserved"] = fmt_gb(torch.cuda.max_memory_reserved())
        res["ms"] = s.elapsed_time(e) / args.iters
        res["grads"] = {n: p.grad.detach().clone()
                        for n, p in mods.named_parameters() if p.grad is not None}
        with torch.no_grad(), ctx:
            res["out"] = fwd().detach()
    except torch.cuda.OutOfMemoryError:
        res["oom"] = True
        res["bwd_peak"] = fmt_gb(torch.cuda.max_memory_allocated())
    finally:
        gc.collect()
        torch.cuda.empty_cache()
    return res


def out_diff(a, b):
    if a is None or b is None or a.shape != b.shape:
        return None
    af, bf = a.float(), b.float()
    mx = (af - bf).abs().max().item()
    ref = bf.abs().max().item()
    return mx, ref, mx / (ref + 1e-30)


def grad_diff(a, b):
    if not a or not b:
        return None
    mx, ref = 0.0, 0.0
    for k in a:
        if k not in b:
            return None
        d = (a[k].float() - b[k].float()).abs().max().item()
        r = a[k].float().abs().max().item()
        mx = max(mx, d)
        ref = max(ref, r)
    return mx, ref, mx / (ref + 1e-30)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    args.we = WE_DEFAULT
    dev = torch.device(args.dev)
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(0)

    print("=" * 78)
    print("bench_enc_ckpt -- encoder gradient checkpointing: cost and benefit")
    print("=" * 78)
    print("torch %s | CUDA %s | %s" % (torch.__version__, torch.version.cuda,
                                       torch.cuda.get_device_name(dev)))
    print("dtype=%s | layers=%d h=%d heads=%d ffn=%d dp=%.2f norm_first=%s"
          % (args.dtype, args.layers, args.h, args.heads, args.ffn, args.dp,
             args.norm_first))
    print("chunk_tokens=%d | warmup=%d iters=%d"
          % (args.chunk_tokens, args.warmup, args.iters))

    # ---- data ----
    t0 = time.time()
    print("\n[data] preparing real label-description features ...")
    if args.dry_data:
        c_input_word, c_word_mask, code_count = synthetic_label_features(args)
        print("   (--dry_data: synthetic)")
    else:
        c_input_word, c_word_mask, code_count = real_label_features(args)
    if args.seg_limit:
        c_input_word = c_input_word[:args.seg_limit]
        c_word_mask = c_word_mask[:args.seg_limit]
        print("   [--seg_limit] first %d descriptions only (%.1f%% of scale)"
              % (args.seg_limit, 100.0 * args.seg_limit / 35684.0))
    c_input_word = c_input_word.to(dev)
    c_word_mask = c_word_mask.to(dev)
    lengths = c_word_mask.sum(-1)
    n_seg, L = c_word_mask.shape
    tot_slot = n_seg * L
    tot_real = int(lengths.sum().item())
    print("   code_count=%d | descriptions=%d | L=%d | token slots=%d | real tokens=%d (%.1f%%)"
          % (code_count, n_seg, L, tot_slot, tot_real, 100.0 * tot_real / tot_slot))
    print("   len min=%d max=%d mean=%.2f | setup time %.1fs"
          % (int(lengths.min()), int(lengths.max()), float(lengths.float().mean()),
             time.time() - t0))
    assert tot_real > 0

    act_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    modes = ["padded", "varlen"] if args.mode == "both" else [args.mode]
    results = {}

    for mode in modes:
        print("\n" + "-" * 78)
        print("### representation: %s" % mode)
        if mode == "varlen" and args.dtype == "fp32":
            _fp32_varlen_note(args)
        print("-" * 78)
        # Rebuild the stack and inputs per mode, with identical weights
        mods = build_stack(mode, args).to(dev).train()
        if mode == "varlen":
            from model.varlen import lengths_to_cu_seqlens
            cu = lengths_to_cu_seqlens(lengths).contiguous()
            x = torch.randn(tot_real, args.h, device=dev, dtype=act_dtype,
                            requires_grad=True)
            mask = None
        else:
            from model.combiner import Transformer_Combiner
            mask = Transformer_Combiner._padding_mask(c_word_mask).to(dev)
            x = torch.randn(n_seg, L, args.h, device=dev, dtype=act_dtype,
                            requires_grad=True)
            cu = None
        print("   input x: %s %s | params %.1f MB"
              % (tuple(x.shape), x.dtype,
                 sum(p.numel() for p in mods.parameters()) * 4 / 1024 ** 2))

        ref_grads = None
        ref_out = None
        for cfg in configs:
            if mode == "padded" and cfg == "sublayer":
                print("   [skip] padded does not support sub-layer granularity")
                continue
            torch.manual_seed(1234)          # identical dropout RNG across configurations
            print("\n   >>> %s / %s" % (mode, cfg), flush=True)
            try:
                r = measure(mode, cfg, mods, x, cu, mask, args)
            except Exception:
                print("   [error]\n" + traceback.format_exc())
                continue
            r["n_seg"] = n_seg
            r["tot_real"] = tot_real
            results[(mode, cfg)] = r
            if r.get("oom"):
                print("   OOM (abandoned) | forward peak %.2f GB | backward peak %.2f GB"
                      % (r.get("fwd_peak", 0), r.get("bwd_peak", 0)))
                torch.cuda.empty_cache()
                time.sleep(2)
                continue
            print("   forward peak %.2f GB | fwd+bwd peak %.2f GB (reserved %.2f GB)"
                  % (r["fwd_peak"], r["bwd_peak"], r["reserved"]))
            print("   fwd+bwd %.1f ms/step | chunks %d" % (r["ms"], r["n_chunks"]))
            if cfg == "none":
                ref_grads = r.pop("grads", None)
                ref_out = r.pop("out", None)
            else:
                od = out_diff(r.pop("out", None), ref_out)
                gd = grad_diff(r.pop("grads", None), ref_grads)
                if od is None:
                    print("   equivalence: not comparable (no none baseline; it usually OOMs)")
                else:
                    print("   output equivalence: max|d|=%.3e  max|ref|=%.3e  rel=%.3e"
                          % od)
                    r["out_rel"] = od[2]
                if gd is None:
                    print("   gradient equivalence: not comparable")
                else:
                    print("   gradient equivalence: max|d|=%.3e  max|ref|=%.3e  rel=%.3e"
                          % gd)
                    r["grad_rel"] = gd[2]
        # Against the baseline
        b = results.get((mode, "none"))
        if b and not b.get("oom"):
            print("\n   ratio to the none baseline:")
            for cfg in configs:
                r = results.get((mode, cfg))
                if not r or r.get("oom"):
                    continue
                print("      %-7s fwd %.2fx | fwd+bwd %.2fx | time %.2fx"
                      % (cfg, r["fwd_peak"] / b["fwd_peak"],
                         r["bwd_peak"] / b["bwd_peak"], r["ms"] / b["ms"]))
            print("   (the time ratio is against none; there is no comparison when none does not fit)")
            # Relationship to the full step time, when a reference is given
            print("   reference: full step times are in study/run_matrix_v2.sh output")

        del mods, x, mask, cu
        gc.collect()
        torch.cuda.empty_cache()

    # ---- summary table ----
    print("\n" + "=" * 78)
    print("summary")
    print("=" * 78)
    hdr = "%-7s %-8s %10s %10s %10s %9s %9s %8s" % (
        "mode", "config", "fwd(GB)", "fwd+bwd(GB)", "ms/step", "out_rel",
        "grad_rel", "chunks")
    print(hdr)
    print("-" * len(hdr))
    for mode in modes:
        for cfg in configs:
            r = results.get((mode, cfg))
            if not r:
                continue
            if r.get("oom"):
                print("%-7s %-8s %10.2f %10.2f %10s %9s %9s %8s"
                      % (mode, cfg, r.get("fwd_peak", 0), r.get("bwd_peak", 0),
                         "OOM", "OOM", "OOM", r.get("n_chunks", "-")))
            else:
                print("%-7s %-8s %10.2f %10.2f %10.1f %9s %9s %8d"
                      % (mode, cfg, r["fwd_peak"], r["bwd_peak"], r["ms"],
                         ("%.2e" % r["out_rel"]) if "out_rel" in r else "ref",
                         ("%.2e" % r["grad_rel"]) if "grad_rel" in r else "ref",
                         r["n_chunks"]))
    print("\ndone.")


if __name__ == "__main__":
    main()
