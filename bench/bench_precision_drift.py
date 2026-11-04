#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""bench_precision_drift.py -- what does bf16 actually perturb?

The paper shows the label side carries most of the bf16 damage and that the effect grows
with label-space size. Why that happens is a mechanism, so this script measures it
directly instead of inferring it:

  A) Perturbation profile: one fp32-trained checkpoint, fp32 forward against bf16 autocast
     forward.
       - relative drift of the label features (mean / p99)
       - relative drift of the text hidden states (control)
       - drift of the decoder logits and probabilities
       - probability drift bucketed by training frequency, plus top-8 rank agreement
     A small drift together with rank inversions on rare codes would explain the pattern:
     micro-F1 barely moves while macro-F1 collapses.

  B) Learned representations: checkpoints of cell 1 (fp32), cell 2 (bf16) and cell 3
     (Transformer).
       - norm and effective rank (entropy of singular values) of the label features
       - shape of the probability distribution: numbers of codes above 0.5

Usage (needs only tens of MB of memory and a few minutes):
    python bench_precision_drift.py --tag 1_msmn_fp32_naive
    python bench_precision_drift.py --compare 1_msmn_fp32_naive 2_msmn_bf16_naive
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.environ.get("ICD_RUNS", os.path.join(HERE, "..", "runs"))


def find_run_dir(tag):
    cands = [d for d in glob.glob(os.path.join(RUNS, "*")) if d.endswith(tag)]
    if not cands:
        raise SystemExit("run directory not found: %s" % tag)
    return cands[0]


def load_model(tag, device):
    """Same convention as dump_predictions.py: same args, datasets and checkpoint loading."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from dump_predictions import load_args, build_datasets, find_checkpoint
    from train_utils import generate_model

    run_dir = find_run_dir(tag)
    ckpt = find_checkpoint(run_dir)
    if ckpt is None:
        raise SystemExit("%s has no usable checkpoint" % tag)
    args = load_args(run_dir, device)
    train_ds, dev_ds, test_ds = build_datasets(args)
    model = generate_model(args, train_ds)
    obj = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = obj if isinstance(obj, torch.nn.Module) else model
    model = model.to(device).eval()
    model.c_input_word = train_ds.c_input_word.to(device)
    model.c_word_mask = train_ds.c_word_mask.to(device)
    return args, train_ds, dev_ds, model


def rel_drift(a, b):
    """Relative drift: mean and p99 of |b-a| / (|a|+eps), a = fp32 reference, b = perturbed."""
    d = (b.float() - a.float()).abs()
    r = d / (a.float().abs() + 1e-6)
    return (float(r.mean()), float(torch.quantile(r.flatten()[: min(r.numel(), 2_000_000)], 0.99)),
            float(d.mean()), float(a.float().abs().mean()))


def _text_hidden(model, iw, wm):
    """Fetch the text hidden states without relying on any method added later."""
    tr = getattr(model.args, "text_repr_final", None)
    return model.encoder(iw, wm, tr)


def _label_feats(model):
    """Label features: use the separate label encoder when there is one (cells 7/8),
    otherwise reuse the text encoder."""
    lenc = getattr(model, "label_text_encoder", None)
    if lenc is None:
        lenc = model.encoder
    lr = getattr(model.args, "label_repr_final", None)
    lh = lenc(model.c_input_word, model.c_word_mask, lr)
    return model.label_encoder(lh, model.c_word_mask)


def forward_parts(model, batch, autocast_dtype=None):
    """One forward pass returning the three intermediates (text hidden, label features,
    decoder logits)."""
    iw, wm = batch[0], batch[1]
    ctx = (torch.autocast("cuda", dtype=autocast_dtype) if autocast_dtype is not None
           else torch.autocast("cuda", enabled=False))
    with torch.no_grad(), ctx:
        th = _text_hidden(model, iw, wm)
        lf = _label_feats(model)
        lg = model.decoder(th, wm, lf)
    return th.detach().float(), lf.detach().float(), lg.detach().float()


def freq_buckets(train_freq):
    """Five buckets: N=1 / 2-5 / 6-20 / 21-100 / >100, as in bucket_metrics.py."""
    f = np.asarray(train_freq)
    edges = [(-1, 1), (1, 5), (5, 20), (20, 100), (100, 10 ** 9)]
    names = ["N=1", "2-5", "6-20", "21-100", ">100"]
    return [(n, (f > lo) & (f <= hi)) for n, (lo, hi) in zip(names, edges)]


def _finish(a, lines):
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("appended to %s" % a.out, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="1_msmn_fp32_naive", help="fp32 checkpoint used for the perturbation profile")
    ap.add_argument("--batches", type=int, default=20, help="how many dev batches (16 notes each)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--compare", nargs="*", default=[], help="only run the representation comparison for these tags")
    ap.add_argument("--out", default=os.path.join(HERE, "out", "precision_drift.log"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dec-chunk", type=int, default=0, help="temporarily shrink the decoder code-axis chunk (useful when sharing a GPU with "
                     "training; chunking is an identity transform)")
    ap.add_argument("--light", action="store_true", help="light mode: only B), from label-feature statistics and a previously dumped "
                     "y_prob.npy; runs no decoder and no backward, so it can share a GPU "
                     "with training")
    ap.add_argument("--skip-grad", action="store_true", help="skip C), the gradient drift; backward needs more memory, so this helps alongside training")
    a = ap.parse_args()
    torch.manual_seed(0)

    lines = []

    def P(s):
        print(s, flush=True)
        lines.append(s)

    # ── B) representation comparison (cheap, so first) ───────────────────────
    tags = a.compare if a.compare else []
    if tags:
        P("=" * 74)
        P("B) learned label representations (same dev batches, model-side statistics only)")
        P("=" * 74)
        P("%-26s %10s %10s %10s %10s" % ("tag", "||label||", "eff. rank", "codes p>0.5", "non-zero"))
        for tag in tags:
            try:
                args, train_ds, dev_ds, model = load_model(tag, a.device)
                lf = _label_feats(model).detach().float().cpu()   # (C*M, D)
                lf2 = lf - lf.mean(0, keepdim=True)
                # Effective rank: exponential of the singular-value entropy (lower = more degenerate)
                s = torch.linalg.svdvals(lf2[: min(lf2.shape[0], 4096)])
                p = (s / s.sum()).clamp_min(1e-12)
                eff_rank = float(torch.exp(-(p * p.log()).sum()))
                # Distribution shape: light mode skips the decoder, which needs ~5 GB in fp32,
                # Read y_prob.npy from the long-tail dump, which covers all eight cells.
                n05 = mx05 = float("nan")
                if a.light:
                    yp = os.path.join(HERE, "..", "longtail", "%s", "y_prob.npy") % tag
                    if os.path.exists(yp):
                        pr = np.load(yp)
                        n05 = float((pr > 0.5).sum(1).mean())
                        mx05 = float((pr > 0.5).sum(1).max())
                    else:
                        n05 = mx05 = float("nan")   # leave empty if the dump is missing rather than run the decoder
                else:
                    dl = DataLoader(dev_ds, batch_size=a.batch_size, shuffle=False,
                                    collate_fn=__import__("data_util").my_collate_fn)
                    batch = next(iter(dl))
                    batch = tuple(x.to(a.device) if torch.is_tensor(x) else x for x in batch)
                    th = _text_hidden(model, batch[0], batch[1])
                    lg = model.decoder(th, batch[1], lf.to(a.device))
                    prob = torch.sigmoid(lg.float())
                    n05 = float((prob > 0.5).sum(1).float().mean())
                    mx05 = float((prob > 0.5).sum(1).float().max())
                P("%-26s %10.3f %10.1f %10.1f %10.1f" % (
                    tag, float(lf.norm(dim=1).mean()), eff_rank, n05, mx05))
            except Exception as e:
                P("%-26s  [warn] failed: %r" % (tag, e))

    # ── A) perturbation profile (skipped in light mode: fp32 decoder needs ~5 GB) ──
    if a.light:
        P("")
        P("(--light: skipping A) perturbation profile and C) gradient drift)")
        _finish(a, lines); return
    P("")
    P("=" * 74)
    P("A) perturbation profile: one fp32 model, fp32 forward vs bf16 autocast (tag=%s)" % a.tag)
    P("=" * 74)
    args, train_ds, dev_ds, model = load_model(a.tag, a.device)
    if a.dec_chunk:
        model.decoder.decoder_config["dec_chunk_codes"] = int(a.dec_chunk)
        P("  (decoder code-axis chunk temporarily set to %d, an identity transform)" % a.dec_chunk)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data_util import my_collate_fn
    from dump_predictions import train_freq

    dl = DataLoader(dev_ds, batch_size=a.batch_size, shuffle=False, collate_fn=my_collate_fn)
    acc = {k: [] for k in ["th", "lf", "lg"]}
    probs_f, probs_b = [], []
    for i, batch in enumerate(dl):
        if i >= a.batches:
            break
        batch = tuple(x.to(a.device) if torch.is_tensor(x) else x for x in batch)
        th_f, lf_f, lg_f = forward_parts(model, batch, None)
        th_b, lf_b, lg_b = forward_parts(model, batch, torch.bfloat16)
        for k, (x, y) in zip(["th", "lf", "lg"], [(th_f, th_b), (lf_f, lf_b), (lg_f, lg_b)]):
            acc[k].append(rel_drift(x, y))
        probs_f.append(torch.sigmoid(lg_f).cpu())
        probs_b.append(torch.sigmoid(lg_b).cpu())
    P("  over %d dev batches (%d notes)" % (len(acc["th"]), len(acc["th"]) * a.batch_size))
    P("  %-14s %12s %12s %12s %12s" % ("tensor", "rel. drift mean", "rel. drift p99", "abs. drift", "reference"))
    for k, name in [("th", "text hidden"), ("lf", "label feats"), ("lg", "decoder logits")]:
        m = np.mean(np.array(acc[k]), axis=0)
        P("  %-14s %12.3e %12.3e %12.3e %12.3e" % (name, m[0], m[1], m[2], m[3]))

    pf = torch.cat(probs_f, 0)
    pb = torch.cat(probs_b, 0)
    P("")
    P("  probability drift by training frequency (rare codes are disturbed most, which is "
      "why macro collapses while micro holds):")
    P("  %-10s %10s %12s %12s %14s" % ("bucket", "codes", "mean|dp|", "mean rel. dp", "top8 agree"))
    tfr = train_freq(train_ds, train_ds.code_count)[0]
    C = pf.shape[1]
    for name, mask in freq_buckets(tfr):
        idx = torch.from_numpy(np.where(mask[:C])[0])
        if idx.numel() == 0:
            continue
        dp = (pb[:, idx] - pf[:, idx]).abs().mean().item()
        rel = ((pb[:, idx] - pf[:, idx]).abs() / (pf[:, idx].abs() + 1e-6)).mean().item()
        # top-8 ranking agreement, per note
        k = 8
        tf8 = pf[:, idx].topk(min(k, idx.numel()), dim=1).indices
        tb8 = pb[:, idx].topk(min(k, idx.numel()), dim=1).indices
        inter = [(len(set(x.tolist()) & set(y.tolist())) / max(1, len(set(x.tolist()))))
                 for x, y in zip(tf8, tb8)]
        P("  %-10s %10d %12.3e %12.3e %14.3f" % (name, int(idx.numel()), dp, rel, float(np.mean(inter))))

    # ── C) gradient drift: same weights and batch, fp32 backward vs bf16 backward ──
    if a.skip_grad:
        P("")
        P("(--skip-grad: skipping C) gradient drift)")
        _finish(a, lines); return

    # A) measures a single bf16 forward at inference, which turns out to be tiny, so the
    # damage must come from training dynamics rather than from rounding. Gradient drift is
    # the observable proxy for those dynamics.
    P("")
    P("=" * 74)
    P("C) gradient drift: same weights and batch, fp32 backward vs bf16 backward")
    P("=" * 74)
    # Training mode is kept because cuDNN's RNN backward requires it, but every dropout is
    # disabled, so both backward passes traverse the same deterministic graph and the only
    # difference is precision.
    for _m in model.modules():
        if isinstance(_m, torch.nn.Dropout):
            _m.p = 0.0
        if isinstance(_m, torch.nn.LSTM):
            _m.dropout = 0.0
    model.train()
    groups = {}
    for i, batch in enumerate(dl):
        if i >= min(3, a.batches):
            break
        batch = tuple(x.to(a.device) if torch.is_tensor(x) else x for x in batch)
        grads = {}
        for name, dt in [("fp32", None), ("bf16", torch.bfloat16)]:
            model.zero_grad(set_to_none=True)
            ctx = (torch.autocast("cuda", dtype=dt) if dt is not None
                   else torch.autocast("cuda", enabled=False))
            with ctx:
                out = model(batch, rdrop=False)
                loss = out["loss"] if isinstance(out, dict) else out
            loss.backward()
            grads[name] = {n: p.grad.detach().float().clone()
                           for n, p in model.named_parameters() if p.grad is not None}
        for n, gf in grads["fp32"].items():
            gb = grads["bf16"].get(n)
            if gb is None:
                continue
            key = ("word_encoder" if "word_encoder" in n else
                   "label_encoder" if "label_encoder" in n else
                   "decoder" if n.startswith("decoder") else
                   "encoder(combiner)" if n.startswith("encoder") else "other")
            d = (gb - gf).norm().item(); r = gf.norm().item()
            groups.setdefault(key, []).append((d, r, gf.numel()))
    P("  %-20s %12s %14s %16s" % ("module group", "||dg||/||g||", "param share", "tensors"))
    tot_p = sum(sum(x[2] for x in v) for v in groups.values()) or 1
    for k, v in sorted(groups.items()):
        rel = np.mean([d / (r + 1e-12) for d, r, _ in v])
        par = sum(x[2] for x in v) / tot_p
        P("  %-20s %12.3e %13.1f%% %16d" % (k, rel, par * 100, len(v)))

    _finish(a, lines)


if __name__ == "__main__":
    main()
