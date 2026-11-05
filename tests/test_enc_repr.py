#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""test_enc_repr.py -- check that VarlenEncoderStack's alternative forwards match ``forward``.

Reference: ``forward`` (unchunked flash varlen).

    forward_chunked   chunk at segment boundaries + checkpointing
                      expected: same kernel as forward, differences only from tile order
    forward_bucketed  bucket by exact length + checkpointing
                      expected: dense SDPA, ~1e-6 in fp32 and ~1e-3 in bf16
"""
import argparse
import os
import sys
import traceback

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("ICD_REPO", os.path.join(HERE, "..", "model", "main"))
BENCH = os.path.abspath(os.path.join(HERE, "..", "bench"))
for p in (REPO, HERE, BENCH):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(REPO)

from bench_enc_ckpt import real_label_features  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--seg", type=int, default=800)
ap.add_argument("--dtype", default="both", choices=["bf16", "fp32", "both"])
ap.add_argument("--layers", type=int, default=4)
ap.add_argument("--h", type=int, default=512)
ap.add_argument("--heads", type=int, default=8)
ap.add_argument("--ffn", type=int, default=2048)
ap.add_argument("--chunk_tokens", type=int, default=4096)
ap.add_argument("--norm_first", action="store_true")
ap.add_argument("--dev", default="cuda:0")
args = ap.parse_args()
args.we = os.environ.get("ICD_WORD_EMBEDDING",
                         "")   # required: word2vec model path
args.label_truncate_length = 30
args.term_count = 4
args.sort_method = "random"

dev = torch.device(args.dev)

#: One entry per comparison made; the exit status is derived from it, so a number that
#: moves out of tolerance fails the run instead of scrolling past as a printed digit.
RESULTS = []


def rel(a, b):
    a, b = a.float(), b.float()
    d = (a - b).abs().max().item()
    r = b.abs().max().item()
    return d, r, d / (r + 1e-30)


def grads(mods):
    return {n: (None if p.grad is None else p.grad.detach().clone())
            for n, p in mods.named_parameters()}


def grel(a, b):
    mx = ref = 0.0
    for k in a:
        if a[k] is None or b[k] is None:
            continue
        mx = max(mx, (a[k].float() - b[k].float()).abs().max().item())
        ref = max(ref, b[k].float().abs().max().item())
    return mx, ref, mx / (ref + 1e-30)


def main():
    from model.varlen import (VarlenEncoderStack, lengths_from_mask,
                              lengths_to_cu_seqlens)
    print("=" * 74)
    print("test_enc_repr: three alternative encoder forwards vs the original forward")
    print("=" * 74)
    c_input_word, c_word_mask, code_count = real_label_features(args)
    c_word_mask = c_word_mask[:args.seg].to(dev)
    lengths = lengths_from_mask(c_word_mask)
    cu = lengths_to_cu_seqlens(lengths).contiguous()
    n_seg, Lmax = c_word_mask.shape
    tot = int(lengths.sum().item())
    max_len = int(lengths.max().item())
    print("descriptions=%d | L=%d | real tokens=%d | zero-length segments=%d"
          % (n_seg, Lmax, tot, int((lengths == 0).sum())))

    for dt in (["bf16", "fp32"] if args.dtype == "both" else [args.dtype]):
        act = torch.bfloat16 if dt == "bf16" else torch.float32
        ctx = torch.autocast("cuda", dtype=torch.bfloat16) if dt == "bf16" \
            else torch.autocast("cuda", enabled=False)
        torch.manual_seed(0)
        stack = VarlenEncoderStack(args.layers, args.h, args.heads, args.ffn,
                                   0.0, "gelu", args.norm_first).to(dev).float().train()
        print("\n" + "-" * 74)
        print("### dtype=%s" % dt)
        print("-" * 74)

        def run(tag):
            for p in stack.parameters():
                p.grad = None
            torch.manual_seed(3)
            x = torch.randn(tot, args.h, device=dev, dtype=act, requires_grad=True)
            xp = torch.zeros(n_seg, Lmax, args.h, device=dev, dtype=act)
            xp[c_word_mask.bool()] = x.detach()
            xp.requires_grad_(True)
            try:
                with ctx:
                    if tag == "forward":
                        out = stack(x, cu, max_len)
                    elif tag == "chunked":
                        out = stack.forward_chunked(x, cu, args.chunk_tokens)
                    elif tag == "bucketed":
                        out = stack.forward_bucketed(x, lengths, args.chunk_tokens)
                    else:
                        out = stack.forward_dense_pad(xp, c_word_mask)
                        out = out[c_word_mask.bool()]     # flatten back to ragged to compare with forward
                loss = out.float().sum()
                loss.backward()
                return (out.detach(),
                        x.grad.detach() if x.grad is not None else None,
                        grads(stack))
            except Exception:
                print("   [%s] raised:\n%s" % (tag, traceback.format_exc()))
                return None

        # The reference depends on precision: forward and forward_chunked both go through
        # flash-attn,
        #    is not a legal configuration in fp32.
        # so bf16 uses forward while fp32 uses dense_pad.
        ref_tag = "forward" if dt == "bf16" else "dense_pad"
        others = ["chunked", "bucketed", "dense_pad"] if dt == "bf16" else ["bucketed"]
        ref = run(ref_tag)
        if ref is None:
            print("   reference %s failed, skipped" % ref_tag)
            RESULTS.append(False)
            continue
        print("   reference %s output max|.| = %.4e" % (ref_tag, ref[0].abs().max().item()))
        for tag in others:
            if tag == ref_tag:
                continue
            got = run(tag)
            if got is None:
                RESULTS.append(False)          # a path that raises is a failure, not a skip
                continue
            od = rel(got[0], ref[0])
            gd = grel(got[2], ref[2])
            # Tolerance per (precision, path). Chunked is the same kernel as the reference,
            # so only tile order can differ; bucketed changes the kernel (dense SDPA vs
            # flash), which is why bf16 needs three more orders of magnitude than fp32.
            limit = TOLERANCE[(dt, tag)]
            ok = (od[2] <= limit) and (gd[2] <= limit)
            RESULTS.append(ok)
            print("   %-9s output max|d|=%.3e rel=%.3e | param grad max|d|=%.3e **rel=%.3e**"
                  "  limit=%.0e  %s"
                  % (tag, od[0], od[2], gd[0], gd[2], limit, "PASS" if ok else "FAIL"))
        del stack
        torch.cuda.empty_cache()


#: rel-tolerance per (dtype, path). These are the bounds implied by Table 2 of the paper:
#: chunking is the same kernel (differences only from tile order), bucketing swaps in dense
#: SDPA, and bf16 carries ~3 extra digits of noise. A run that exceeds them is a regression.
TOLERANCE = {
    ("bf16", "chunked"): 1e-4,
    ("bf16", "dense_pad"): 1e-2,
    ("bf16", "bucketed"): 1e-2,
    ("fp32", "bucketed"): 1e-4,
}


if __name__ == "__main__":
    main()
    n_ok = sum(RESULTS)
    print()
    print("=" * 74)
    print("summary: %d/%d comparisons within tolerance" % (n_ok, len(RESULTS)))
    print("=" * 74)
    sys.exit(0 if RESULTS and n_ok == len(RESULTS) else 1)
