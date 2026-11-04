#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""bench_bucketed.py -- validate the length-bucketed representation.

The problem
-----------
The label branch feeds 35,684 variable-length descriptions (1 to 30 words, mean 5.61) into a
four-layer Transformer. There are three ways to feed the same input:

  padded    pad to 30  ->  1,070,520 slots holding 200,353 real tokens (5.34x waste), and a
            mask is required
  varlen    ragged (cu_seqlens), no waste and no mask, but flash-attn ships fp16/bf16 kernels
            only, so fp32 is impossible
  bucketed  group by exact length: all L=1 together, all L=2 together, and so on. Uniform
            within a bucket, so no padding, no mask, and plain SDPA -- fp32 works

Three questions
---------------
  1. Equivalence: with dropout off, do bucketed and padded (and the upstream
     nn.TransformerEncoder) agree in outputs and gradients?
  2. Memory: what are the full-scale forward and forward+backward peaks under bucketed?
  3. Time: milliseconds per step in bf16 and, the point of this check, in fp32 -- only
     bucketed can give the Transformer label branch in fp32 at all.

Criteria
--------
  equivalence: relative error <= 1e-5 (fp32 should be far below; bf16 around 1e-3, the same
               order as the varlen and chunking paths)
  feasibility: under fp32 the label branch must be <= 1.2 s/step and <= 8 GB

Usage
-----
    export ICD_DATA_DIR=/path/to/mimic3_caml
    export ICD_WORD_EMBEDDING=/path/to/word2vec_sg0_100.model
    python -u bench/bench_bucketed.py --mode both --dtype both

Notes
-----
* All four representations share one set of weights (the VarlenEncoderLayer weights are
  copied into nn.TransformerEncoderLayer), otherwise the comparison means nothing.
* The equivalence test fixes dropout=0, otherwise the difference reflects dropout mask
  ordering rather than numerical error.
"""
import argparse
import gc
import os
import sys
import time
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("ICD_REPO", os.path.join(HERE, "..", "model", "main"))
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (REPO, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(REPO)

from bench_enc_ckpt import (WE_DEFAULT, fmt_gb,  # noqa: E402
                            real_label_features, synthetic_label_features)


# --------------------------------------------------------------------------- #
# arguments
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="both", choices=["equiv", "perf", "both"])
    p.add_argument("--dtype", default="both", choices=["bf16", "fp32", "both"])
    p.add_argument("--sdpa_backend", default="auto",
                   choices=["auto", "efficient", "flash", "math"],
                   help="force the dense path's SDPA backend (auto picks flash under bf16 with no mask)")
    p.add_argument("--reprs", default="padded,bucketed,varlen",
                   help="representations to time (varlen is only valid under bf16)")
    p.add_argument("--equiv_seg", type=int, default=1500,
                   help="number of descriptions for the equivalence test (padded must fit, so not full scale)")
    p.add_argument("--chunk_tokens", type=int, default=16384,
                   help="max real tokens per sub-batch")
    p.add_argument("--iters", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--dp", type=float, default=0.1, help="dropout for the timing runs")
    # Matches run_matrix_real.sh and Table 2 of the MSMN paper
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--h", type=int, default=512)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--ffn", type=int, default=2048)
    p.add_argument("--norm_first", action="store_true")
    p.add_argument("--label_truncate_length", type=int, default=30)
    p.add_argument("--term_count", type=int, default=4)
    p.add_argument("--sort_method", default="random")
    p.add_argument("--dev", default="cuda:0")
    p.add_argument("--n_seg", type=int, default=35684, help="number of descriptions under --dry_data")
    p.add_argument("--dry_data", action="store_true")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# One set of weights, four forwards
# --------------------------------------------------------------------------- #
def build_vlayers(args, seed=0, dp=None):
    """The shared-weight layer: standard q/k/v/out plus FFN and two LayerNorms."""
    from model.varlen import VarlenEncoderLayer
    torch.manual_seed(seed)
    d = args.dp if dp is None else dp
    return nn.ModuleList([
        VarlenEncoderLayer(args.h, args.heads, args.ffn, d, "gelu", args.norm_first)
        for _ in range(args.layers)])


def build_nn_layers(args, vlayers, seed=0, dp=None):
    """The upstream path: nn.TransformerEncoderLayer with the vlayers weights copied in."""
    torch.manual_seed(seed)
    d = args.dp if dp is None else dp
    layers = nn.ModuleList([
        nn.TransformerEncoderLayer(d_model=args.h, nhead=args.heads,
                                   dim_feedforward=args.ffn, dropout=d,
                                   activation="gelu", batch_first=True,
                                   norm_first=args.norm_first)
        for _ in range(args.layers)])
    H = args.h
    with torch.no_grad():
        for vl, nl in zip(vlayers, layers):
            nl.self_attn.in_proj_weight.copy_(
                torch.cat([vl.q_proj.weight, vl.k_proj.weight, vl.v_proj.weight]))
            nl.self_attn.in_proj_bias.copy_(
                torch.cat([vl.q_proj.bias, vl.k_proj.bias, vl.v_proj.bias]))
            nl.self_attn.out_proj.weight.copy_(vl.out_proj.weight)
            nl.self_attn.out_proj.bias.copy_(vl.out_proj.bias)
            nl.linear1.weight.copy_(vl.linear1.weight)
            nl.linear1.bias.copy_(vl.linear1.bias)
            nl.linear2.weight.copy_(vl.linear2.weight)
            nl.linear2.bias.copy_(vl.linear2.bias)
            nl.norm1.weight.copy_(vl.norm1.weight)
            nl.norm1.bias.copy_(vl.norm1.bias)
            nl.norm2.weight.copy_(vl.norm2.weight)
            nl.norm2.bias.copy_(vl.norm2.bias)
    return layers


#: Forced SDPA backend (None = automatic). Under bf16 with no mask the automatic choice is
#: the flash kernel, so a non-flash baseline must request EFFICIENT or MATH explicitly.
_SDPA_BACKEND = None


def _sdpa(q, k, v, attn_mask):
    if _SDPA_BACKEND is None:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    from torch.nn.attention import sdpa_kernel
    with sdpa_kernel(_SDPA_BACKEND):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)


def sdpa_attn(layer, x, attn_mask=None):
    """x: (B, L, H) -> (B, L, H), scaled by SDPA's default 1/sqrt(head_dim), as flash-attn."""
    B, L, _ = x.shape
    nh, hd = layer.nhead, layer.head_dim
    q = layer.q_proj(x).view(B, L, nh, hd).transpose(1, 2)
    k = layer.k_proj(x).view(B, L, nh, hd).transpose(1, 2)
    v = layer.v_proj(x).view(B, L, nh, hd).transpose(1, 2)
    a = _sdpa(q, k, v, attn_mask)
    return layer.out_proj(a.transpose(1, 2).reshape(B, L, layer.d_model))


def dense_layer_forward(layer, x, attn_mask=None):
    """Same structure as VarlenEncoderLayer.forward, with attention swapped for dense SDPA."""
    if layer.norm_first:
        x = x + layer.dropout1(sdpa_attn(layer, layer.norm1(x), attn_mask))
        x = x + layer.dropout2(layer._ff(layer.norm2(x)))
    else:
        x = layer.norm1(x + layer.dropout1(sdpa_attn(layer, x, attn_mask)))
        x = layer.norm2(x + layer.dropout2(layer._ff(x)))
    return x


# --------------------------------------------------------------------------- #
# The three forwards
# --------------------------------------------------------------------------- #
def padded_forward(vlayers, x_pad, word_mask, chunk_rows=None):
    """(N, Lmax, H) with a padding mask; chunk_rows splits it by row when given."""
    Lmax = x_pad.shape[1]
    key = word_mask.bool()[:, None, None, :]          # SDPA: True = attended

    def run(x, k):
        for lay in vlayers:
            x = dense_layer_forward(lay, x, k)
        return x

    if chunk_rows is None:
        out = run(x_pad, key)
    else:
        outs = []
        for i in range(0, x_pad.shape[0], chunk_rows):
            xc = x_pad[i:i + chunk_rows].contiguous()
            kc = key[i:i + chunk_rows]
            outs.append(torch.utils.checkpoint.checkpoint(
                run, xc, kc, use_reentrant=False))
        out = torch.cat(outs, 0)
    return out * word_mask.unsqueeze(-1).to(out.dtype)   # matches upstream zero_pad_output


def nn_padded_forward(nn_layers, x_pad, word_mask):
    """The upstream implementation (nn.TransformerEncoder + src_key_padding_mask), reference only."""
    m = word_mask == 0
    if bool(m.all(dim=1).any()):                      # keep one position open on fully padded rows
        m = m.clone()
        m[m.all(dim=1), 0] = False
    x = x_pad
    for lay in nn_layers:
        x = lay(x, src_key_padding_mask=m)
    return x * word_mask.unsqueeze(-1).to(x.dtype)


def make_batches(lengths, chunk_tokens):
    """Bucket by length -> [(L, idx_tensor)], each sub-batch <= chunk_tokens."""
    by_len = {}
    for i, L in enumerate(lengths.tolist()):
        if L > 0:
            by_len.setdefault(L, []).append(i)
    batches = []
    for L in sorted(by_len):
        idxs = by_len[L]
        per = max(1, chunk_tokens // L)
        for s in range(0, len(idxs), per):
            batches.append((L, torch.tensor(idxs[s:s + per], dtype=torch.long)))
    return batches


def bucketed_forward(vlayers, x_flat, batches, starts, scatter=None, ckpt=False):
    """x_flat: (SL, H) ragged input; batches come from make_batches.

    When scatter is a zeroed (N, Lmax, H) tensor the results are written back into the
    padded layout, for the equivalence comparison; otherwise the per-subbatch outputs are
    returned, which avoids allocating another large tensor during timing.
    """
    outs = []
    for L, idx in batches:
        gidx = starts[idx].unsqueeze(1) + torch.arange(L, device=x_flat.device)
        xb = x_flat[gidx]                       # (n, L, H), all real tokens
        if ckpt:
            xb = torch.utils.checkpoint.checkpoint(
                _run_dense, vlayers, xb, use_reentrant=False)
        else:
            for lay in vlayers:
                xb = dense_layer_forward(lay, xb, None)
        outs.append((L, idx, xb))
    if scatter is not None:
        Lmax = scatter.shape[1]
        for L, idx, xb in outs:
            scatter.index_put_((idx.to(scatter.device),),
                               F.pad(xb, (0, 0, 0, Lmax - L)))
        return scatter
    return [xb for _, _, xb in outs]          # timing mode: tensors only, no reassembly


def _run_dense(vlayers, x):
    for lay in vlayers:
        x = dense_layer_forward(lay, x, None)
    return x


def varlen_forward(vlayers, x_flat, cu, max_len, batches=None):
    """flash-attn varlen; batches splits it at segment boundaries when given."""
    if batches is None:
        x = x_flat
        for lay in vlayers:
            x = lay(x, cu, max_len)
        return x
    ys = []
    for _, idx in batches:                    # idx holds segment ids; see make_varlen_batches
        i0, i1 = int(idx[0]), int(idx[-1]) + 1
        base = int(cu[i0])
        xc = x_flat[base:int(cu[i1])].contiguous()
        cuc = (cu[i0:i1 + 1] - base).contiguous()
        ml = int((cu[i0 + 1:i1 + 1] - cu[i0:i1]).max())
        ys.append(torch.utils.checkpoint.checkpoint(
            _run_varlen, vlayers, xc, cuc, ml, use_reentrant=False))
    return torch.cat(ys, 0)


def _run_varlen(vlayers, x, cu, max_len):
    for lay in vlayers:
        x = lay(x, cu, max_len)
    return x


def make_varlen_batches(lengths, chunk_tokens):
    """Split at segment boundaries -> [(None, segment_id_tensor)]."""
    lens = lengths.tolist()
    groups, cur, tok = [], [], 0
    for i, L in enumerate(lens):
        if cur and tok + L > chunk_tokens:
            groups.append(cur)
            cur, tok = [], 0
        cur.append(i)
        tok += L
    if cur:
        groups.append(cur)
    return [(None, torch.tensor(g, dtype=torch.long)) for g in groups]


# --------------------------------------------------------------------------- #
# Gradient collection (split the nn layer's in_proj_weight back into q/k/v for comparison)
# --------------------------------------------------------------------------- #
def v_grads(vlayers):
    d = {}
    for i, l in enumerate(vlayers):
        for n, p in l.named_parameters():
            d["L%d.%s" % (i, n)] = None if p.grad is None else p.grad.detach()
    return d


def nn_grads_as_v(nn_layers, H):
    d = {}
    for i, l in enumerate(nn_layers):
        w = l.self_attn.in_proj_weight.grad
        b = l.self_attn.in_proj_bias.grad
        for j, nm in enumerate(("q_proj", "k_proj", "v_proj")):
            d["L%d.%s.weight" % (i, nm)] = w[j * H:(j + 1) * H]
            d["L%d.%s.bias" % (i, nm)] = b[j * H:(j + 1) * H]
        d["L%d.out_proj.weight" % i] = l.self_attn.out_proj.weight.grad
        d["L%d.out_proj.bias" % i] = l.self_attn.out_proj.bias.grad
        for nm in ("linear1", "linear2"):
            d["L%d.%s.weight" % (i, nm)] = getattr(l, nm).weight.grad
            d["L%d.%s.bias" % (i, nm)] = getattr(l, nm).bias.grad
        for nm in ("norm1", "norm2"):
            d["L%d.%s.weight" % (i, nm)] = getattr(l, nm).weight.grad
            d["L%d.%s.bias" % (i, nm)] = getattr(l, nm).bias.grad
    return d


def diff(a, b, name):
    """Return (max|d|, max|b|, rel)."""
    if a is None or b is None:
        return None
    if isinstance(a, dict):
        if set(a) != set(b):
            print("   [%s] key mismatch" % name)
            return None
        mx = ref = 0.0
        worst = None
        for k in a:
            if a[k] is None or b[k] is None:
                continue
            af, bf = a[k].float(), b[k].float()
            d = (af - bf).abs().max().item()
            if d > mx:
                mx, worst = d, k
            ref = max(ref, bf.abs().max().item())
        return mx, ref, mx / (ref + 1e-30), worst
    af, bf = a.float(), b.float()
    mx = (af - bf).abs().max().item()
    ref = bf.abs().max().item()
    return mx, ref, mx / (ref + 1e-30), None


# --------------------------------------------------------------------------- #
# Equivalence test
# --------------------------------------------------------------------------- #
def _segment_starts(lengths, dev):
    lengths = lengths.to(dev)
    z = torch.zeros(1, dtype=torch.long, device=dev)
    return torch.cat([z, torch.cumsum(lengths, 0)[:-1]]).long()


def run_equiv(args, c_word_mask, lengths, dtype):
    """Reference = padded_dense: same weights, same layer code, differing only in representation.

    padded_nn (the upstream nn.TransformerEncoder) is an extra row: its difference from
    padded_dense reflects a different attention kernel, not a different representation, and
    confirms the dense implementation is correct.
    """
    print("\n" + "=" * 78)
    print("### equivalence | dtype=%s | descriptions=%d | dropout=0" % (dtype, len(lengths)))
    print("=" * 78)
    dev = torch.device(args.dev)
    n_seg, Lmax = c_word_mask.shape
    mask_dev = c_word_mask.to(dev)
    mb = mask_dev.bool()
    lengths = lengths.to(dev)

    vlayers = build_vlayers(args, dp=0.0).to(dev).float().train()
    nn_layers = build_nn_layers(args, vlayers, dp=0.0).to(dev).float().train()
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if dtype == "bf16" \
        else torch.autocast("cuda", enabled=False)

    cu = None
    if "varlen" in args.reprs:
        from model.varlen import lengths_to_cu_seqlens
        cu = lengths_to_cu_seqlens(lengths).contiguous()
    batches_b = make_batches(lengths.cpu(), args.chunk_tokens)
    starts = _segment_starts(lengths, dev)
    max_len = int(lengths.max().item())

    torch.manual_seed(7)
    x_pad = torch.randn(n_seg, Lmax, args.h, device=dev, dtype=torch.float32,
                        requires_grad=True)
    x_flat0 = x_pad.detach()[mb].contiguous()
    # Fixed random weights keep the input gradient well conditioned; with sum() and LayerNorm
    # it cancels to ~1e-7 and a relative error means nothing
    torch.manual_seed(99)
    W_pad = torch.randn(n_seg, Lmax, args.h, device=dev, dtype=torch.float32)
    W_flat = W_pad[mb].contiguous()

    def collect(name):
        for l in list(vlayers) + list(nn_layers):
            for p in l.parameters():
                p.grad = None
        x_pad.grad = None
        torch.manual_seed(7)
        inp = x_pad if name in ("padded_dense", "padded_nn") \
            else x_flat0.clone().requires_grad_(True)
        with ctx:
            if name == "padded_dense":
                out = padded_forward(vlayers, inp, mask_dev)
                loss = (out.float() * W_pad * mask_dev.unsqueeze(-1)).sum()
            elif name == "padded_nn":
                out = nn_padded_forward(nn_layers, inp, mask_dev)
                loss = (out.float() * W_pad * mask_dev.unsqueeze(-1)).sum()
            elif name == "bucketed":
                out = bucketed_forward(vlayers, inp, batches_b, starts,
                                       torch.zeros(n_seg, Lmax, args.h,
                                                   device=dev, dtype=torch.float32))
                loss = (out.float() * W_pad * mask_dev.unsqueeze(-1)).sum()
            else:
                out = varlen_forward(vlayers, inp, cu, max_len)
                loss = (out.float() * W_flat).sum()
        loss.backward()
        grads = nn_grads_as_v(nn_layers, args.h) if name == "padded_nn" \
            else v_grads(vlayers)
        grads = {k: (None if v is None else v.detach().clone())
                 for k, v in grads.items()}
        if name == "varlen":                      # (SL, H) -> padded layout
            gp = torch.zeros(n_seg, Lmax, args.h, device=dev, dtype=torch.float32)
            out_pad = gp.index_put((mb,), out.detach())
        else:
            out_pad = out.detach()
        xg = inp.grad.detach()
        xg = xg[mb] if name in ("padded_dense", "padded_nn") else xg
        return out_pad, grads, xg

    order = ["padded_dense", "padded_nn", "bucketed"] + \
            (["varlen"] if "varlen" in args.reprs else [])
    ref = None
    results = {}
    for name in order:
        if name == "varlen" and dtype == "fp32":
            print("\n   >>> varlen/fp32 -- skipped (flash-attn has half-precision kernels only)")
            continue
        try:
            out, grads, xg = collect(name)
        except Exception:
            print("\n   >>> %s raised:\n%s" % (name, traceback.format_exc()))
            continue
        om = (out.float() * mask_dev.unsqueeze(-1))[mb]
        if ref is None:
            ref = (out, grads, xg)
            print("\n   >>> %s  <- reference   output max|.|=%.4e"
                  % (name, om.abs().max().item()))
            results[name] = ((0.0, 0.0, 0.0, None), (0.0, 0.0, 0.0, None),
                             (0.0, 0.0, 0.0, None))
            continue
        od = diff(om, (ref[0].float() * mask_dev.unsqueeze(-1))[mb], "out")
        gd = diff(grads, ref[1], "grad")
        xd = diff(xg, ref[2], "xgrad")
        results[name] = (od, gd, xd)
        print("\n   >>> %s" % name)
        for tag, r in (("output", od), ("param grad", gd), ("input grad", xd)):
            if r is None:
                print("      %-10s not comparable" % tag)
            else:
                print("      %-6s max|Δ|=%.3e  max|ref|=%.3e  **rel=%.3e**%s"
                      % (tag, r[0], r[1], r[2],
                         "" if r[3] is None else "  (worst param %s)" % r[3]))
        del out, grads, xg
        gc.collect()
        torch.cuda.empty_cache()

    del vlayers, nn_layers, x_pad, x_flat0, ref
    gc.collect()
    torch.cuda.empty_cache()
    return results


# --------------------------------------------------------------------------- #
# Timing
# --------------------------------------------------------------------------- #
def measure(args, name, vlayers, fwd, dtype):
    dev = torch.device(args.dev)
    act = torch.bfloat16 if dtype == "bf16" else torch.float32
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if dtype == "bf16" \
        else torch.autocast("cuda", enabled=False)
    res = {"name": name, "dtype": dtype, "oom": False}

    def step():
        for p in vlayers.parameters():
            p.grad = None
        with ctx:
            loss = sum(t.float().sum() for t in fwd())
        loss.backward()

    # --- fwd only ---
    try:
        with torch.no_grad(), ctx:
            sum(t.float().sum() for t in fwd())
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
        print("   %-9s %-5s  OOM in forward (peak lower bound %.2f GB)"
              % (name, dtype, res["fwd_peak"]))
        return res
    except Exception:
        print("   %-9s %-5s  raised:\n%s" % (name, dtype, traceback.format_exc()))
        res["oom"] = True
        return res

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
    except torch.cuda.OutOfMemoryError:
        res["oom"] = True
        res["bwd_peak"] = fmt_gb(torch.cuda.max_memory_allocated())
    except Exception:
        print("   %-9s %-5s  backward raised:\n%s" % (name, dtype, traceback.format_exc()))
        res["oom"] = True
    finally:
        gc.collect()
        torch.cuda.empty_cache()

    if not res["oom"]:
        print("   %-9s %-5s  fwd %5.2f GB | fwd+bwd %5.2f GB | %7.1f ms/step"
              % (name, dtype, res["fwd_peak"], res["bwd_peak"], res["ms"]))
    else:
        print("   %-9s %-5s  OOM in fwd+bwd (peak lower bound %.2f GB)"
              % (name, dtype, res.get("bwd_peak", 0)))
    return res


def run_perf(args, c_word_mask, lengths, dtypes, reprs):
    print("\n" + "=" * 78)
    print("### timing (full scale) | dropout=%.2f | chunk_tokens=%d" % (args.dp, args.chunk_tokens))
    print("=" * 78)
    dev = torch.device(args.dev)
    n_seg, Lmax = c_word_mask.shape
    tot_real = int(lengths.sum().item())
    print("   descriptions=%d | real tokens=%d | slots=%d | waste %.2fx"
          % (n_seg, tot_real, n_seg * Lmax, n_seg * Lmax / tot_real))
    mask_dev = c_word_mask.to(dev)
    batches_b = make_batches(lengths, args.chunk_tokens)
    batches_v = make_varlen_batches(lengths, args.chunk_tokens)
    cu = None
    if "varlen" in reprs:
        from model.varlen import lengths_to_cu_seqlens
        cu = lengths_to_cu_seqlens(lengths).to(dev).contiguous()
    max_len = int(lengths.max().item())
    chunk_rows = max(1, args.chunk_tokens // Lmax)
    starts = _segment_starts(lengths, dev)
    results = []

    for dtype in dtypes:
        act = torch.bfloat16 if dtype == "bf16" else torch.float32
        for name in reprs:
            if name == "varlen" and dtype == "fp32":
                print("   %-9s %-5s  skipped (flash-attn has half-precision kernels only)" % (name, dtype))
                continue
            vlayers = build_vlayers(args, dp=args.dp).to(dev).float().train()
            x_pad = x_flat = None
            torch.manual_seed(11)
            if name == "padded":
                x_pad = torch.randn(n_seg, Lmax, args.h, device=dev, dtype=act,
                                    requires_grad=True)
                fwd = lambda: [padded_forward(vlayers, x_pad, mask_dev, chunk_rows)]
            elif name == "bucketed":
                x_flat = torch.randn(tot_real, args.h, device=dev, dtype=act,
                                    requires_grad=True)
                fwd = lambda: bucketed_forward(vlayers, x_flat, batches_b,
                                               starts, ckpt=True)
            else:
                x_flat = torch.randn(tot_real, args.h, device=dev, dtype=act,
                                    requires_grad=True)
                fwd = lambda: [varlen_forward(vlayers, x_flat, cu, max_len, batches_v)]
            r = measure(args, name, vlayers, fwd, dtype)
            r["n_batches"] = len(batches_b) if name == "bucketed" else (
                len(batches_v) if name == "varlen" else
                (n_seg + chunk_rows - 1) // chunk_rows)
            results.append(r)
            # max_memory_allocated is absolute: leftover input tensors inflate the next config
            del vlayers, fwd, x_pad, x_flat
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(1)
    return results


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    global _SDPA_BACKEND
    if args.sdpa_backend != "auto":
        from torch.nn.attention import SDPBackend
        _SDPA_BACKEND = {"efficient": SDPBackend.EFFICIENT_ATTENTION,
                         "flash": SDPBackend.FLASH_ATTENTION,
                         "math": SDPBackend.MATH}[args.sdpa_backend]
        print("[sdpa] forced backend = %s" % args.sdpa_backend)
    args.we = WE_DEFAULT
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(0)
    dev = torch.device(args.dev)

    print("=" * 78)
    print("bench_bucketed -- length-bucketed representation check")
    print("=" * 78)
    print("torch %s | %s" % (torch.__version__, torch.cuda.get_device_name(dev)))
    print("layers=%d h=%d heads=%d ffn=%d norm_first=%s"
          % (args.layers, args.h, args.heads, args.ffn, args.norm_first))
    print("SDPA: flash=%s mem_efficient=%s math=%s"
          % (torch.backends.cuda.flash_sdp_enabled(),
             torch.backends.cuda.mem_efficient_sdp_enabled(),
             torch.backends.cuda.math_sdp_enabled()))

    t0 = time.time()
    print("\n[data] real label-description features ...")
    if args.dry_data:
        c_input_word, c_word_mask, code_count = synthetic_label_features(args)
        print("   (--dry_data: synthetic)")
    else:
        c_input_word, c_word_mask, code_count = real_label_features(args)
    lengths_all = c_word_mask.sum(-1)
    print("   code_count=%d | descriptions=%d | L=%d | real tokens=%d (%.1f%%) | len min=%d max=%d mean=%.2f"
          % (code_count, c_word_mask.shape[0], c_word_mask.shape[1],
             int(lengths_all.sum()), 100.0 * float(lengths_all.float().mean()) / c_word_mask.shape[1],
             int(lengths_all.min()), int(lengths_all.max()), float(lengths_all.float().mean())))
    # Bucket composition
    from collections import Counter
    cnt = Counter(int(x) for x in lengths_all.tolist() if x > 0)
    print("   buckets (distinct lengths)=%d | zero-length segments=%d | five largest: %s"
          % (len(cnt), int((lengths_all == 0).sum()),
             sorted(cnt.items(), key=lambda kv: -kv[1])[:5]))
    print("   setup time %.1fs" % (time.time() - t0))

    dtypes = ["bf16", "fp32"] if args.dtype == "both" else [args.dtype]
    reprs = [s.strip() for s in args.reprs.split(",") if s.strip()]

    equiv = {}
    if args.mode in ("equiv", "both"):
        n = min(args.equiv_seg, c_word_mask.shape[0])
        sub_mask = c_word_mask[:n].contiguous()
        sub_len = lengths_all[:n].contiguous()
        print("\n[equivalence] first %d descriptions only (padded must fit)" % n)
        for dt in dtypes:
            equiv[dt] = run_equiv(args, sub_mask, sub_len, dt)

    perf = []
    if args.mode in ("perf", "both"):
        perf = run_perf(args, c_word_mask, lengths_all, dtypes, reprs)

    # ---- summary ----
    print("\n" + "=" * 78)
    print("summary")
    print("=" * 78)
    if equiv:
        print("\n[equivalence] relative error (reference = upstream padded, dropout=0)")
        hdr = "%-6s %-13s %12s %12s %12s" % ("dtype", "repr", "output", "param grad", "input grad")
        print(hdr)
        print("-" * len(hdr))
        for dt in dtypes:
            for name, (od, gd, xd) in (equiv.get(dt) or {}).items():
                print("%-6s %-13s %12s %12s %12s"
                      % (dt, name,
                         "n/a" if od is None else "%.3e" % od[2],
                         "n/a" if gd is None else "%.3e" % gd[2],
                         "n/a" if xd is None else "%.3e" % xd[2]))
    if perf:
        print("\n[timing] full scale (%d descriptions / %d real tokens)"
              % (c_word_mask.shape[0], int(lengths_all.sum())))
        hdr = "%-9s %-6s %10s %12s %10s %8s" % (
            "repr", "dtype", "fwd(GB)", "fwd+bwd(GB)", "ms/step", "sub-batches")
        print(hdr)
        print("-" * len(hdr))
        for r in perf:
            if r["oom"]:
                print("%-9s %-6s %10.2f %12s %10s %8s"
                      % (r["name"], r["dtype"], r.get("fwd_peak", 0), "OOM", "-",
                         r.get("n_batches", "-")))
            else:
                print("%-9s %-6s %10.2f %12.2f %10.1f %8d"
                      % (r["name"], r["dtype"], r["fwd_peak"], r["bwd_peak"],
                         r["ms"], r["n_batches"]))
    print("\ndone.")


if __name__ == "__main__":
    main()
