#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""dump_predictions.py -- export the matrices the long-tail analysis needs from a trained
checkpoint.

bucket_metrics.py needs four .npy files, and this script produces them:
    y_true.npy      (n, C) uint8     test 0/1 labels
    y_prob.npy      (n, C) float32   unthresholded probabilities
    y_pred.npy      (n, C) uint8     binarised with the threshold selected on dev
    train_freq.npy  (C,)   int32     documents per code in TRAINING (the bucketing input)
    meta.json                        threshold, shapes, source run, recomputed metrics

The dev matrices are exported too, so the threshold can be recomputed and its sensitivity
checked.

Conventions match training and evaluation exactly, by reusing the repository's own functions:
  * threshold: find_threshold_micro(dev_prob, dev_y), the same logic as training
  * metrics:   all_metrics(yhat, y, yhat_raw=prob), the same logic as the paper
  * bf16:      predict() handles autocast internally (the varlen path breaks in fp32)
  * binarise with prob > thr, matching eval_func rather than predict(), so the recomputed
    metrics line up with metric_log

Self-check: the recomputed test metrics are compared against Best_Test in the run's
metric_log, and a mismatch is reported loudly.

Needs a GPU; avoid running it while the study is training.

Usage:
    python dump_predictions.py --run-dir RUNS/<cell> --out OUT/longtail
    python dump_predictions.py --all --runs RUNS --out OUT/longtail
    python dump_predictions.py --all --runs RUNS --out OUT/longtail --dry-run
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader


# The repository must be resolved BEFORE torch.load. The hybrid cells' checkpoints pickle the
# whole nn.Module rather than a state_dict, so deserialising imports the IcdModel class from
# sys.path. Resolving the wrong repository loads the main-variant class and silently computes
# the wrong forward pass (measured: logits median pushed to -30, AUC 0.9453 -> 0.4743).
# Resolution order: --repo > ICD_REPO > cwd (if it looks like a repo) > model/main. The
# resolved repository and the class file it provided are printed.
def _looks_like_repo(p):
    return os.path.isfile(os.path.join(p, "model", "icd_model.py"))


def resolve_repo(cli_repo=None):
    for cand, why in ((cli_repo, "--repo"), (os.environ.get("ICD_REPO"), "ICD_REPO"),
                      (os.getcwd(), "cwd")):
        if cand and _looks_like_repo(cand):
            return os.path.abspath(cand), why
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model", "main"), "default (main)"


def set_repo(repo, why):
    """Put the repository on sys.path and chdir; must run before torch.load."""
    global REPO, REPO_WHY
    REPO, REPO_WHY = os.path.abspath(repo), why
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    os.chdir(REPO)
    return REPO


set_repo(*resolve_repo(None))             # initial value from env/cwd

WE = os.environ.get("ICD_WORD_EMBEDDING", "")   # required: word2vec model path


# --------------------------------------------------------------------------- #
def load_args(run_dir, device):
    """Turn a run's args.json into usable args, filling in device and environment fields."""
    with open(os.path.join(run_dir, "args.json"), encoding="utf-8") as f:
        d = json.load(f)
    ns = argparse.Namespace(**d)
    ns.device = device
    ns.n_gpu = 1
    ns.word_embedding_path = ns.word_embedding_path or WE
    # resolve_reprs reruns inside generate_model, so the raw CLI fields suffice
    return ns


def build_datasets(args):
    """Build the dataset the same way training does; version and mode pick the json."""
    from data_util import MimicFullDataset
    common = dict(vocab_path=args.word_embedding_path,
                  truncate_length=args.truncate_length,
                  label_truncate_length=args.label_truncate_length,
                  term_count=args.term_count, sort_method=args.sort_method)
    train_ds = MimicFullDataset(args.version, "train", **common)
    dev_ds = MimicFullDataset(args.version, "dev", **common)
    test_ds = MimicFullDataset(args.version, "test", **common)
    return train_ds, dev_ds, test_ds


def find_checkpoint(run_dir):
    """Prefer best_epoch.pth, else the highest-numbered epoch{N}.pth."""
    best = os.path.join(run_dir, "best_epoch.pth")
    if os.path.exists(best):
        return best
    eps = []
    for fn in os.listdir(run_dir):
        if fn.startswith("epoch") and fn.endswith(".pth"):
            try:
                eps.append((int(fn[5:-4]), fn))
            except ValueError:
                pass
    if not eps:
        return None
    return os.path.join(run_dir, sorted(eps)[-1][1])


def train_freq(train_ds, n_labels):
    """Documents per code in TRAINING, the bucketing input. Read from the json, no tokenising."""
    from constant import MIMIC_3_DIR
    path = os.path.join(MIMIC_3_DIR, "%s_train.json" % train_ds.version)
    freq = np.zeros(n_labels, dtype=np.int64)
    unknown = 0
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    c2ind = train_ds.c2ind
    for rec in data:
        labs = rec["LABELS"]
        labs = labs.split(";") if isinstance(labs, str) else list(labs)
        for c in set(labs):
            if c in c2ind:
                freq[c2ind[c]] += 1
            elif c:
                unknown += 1
    return freq, len(data), unknown


def best_test_from_metric_log(run_dir):
    """Read the Best_Test block of metric_log, for the self-check."""
    p = os.path.join(run_dir, "metric_log")
    if not os.path.exists(p):
        return None, None
    keys = ["MACRO-auc", "MICRO-auc", "MACRO-f1", "MICRO-f1",
            "P@5", "P@8", "P@15", "R@5", "R@8", "R@15"]
    lines = [l.strip() for l in open(p, encoding="utf-8")]
    for i, l in enumerate(lines):
        if l.startswith("Best_Test_Epoch"):
            ep = int(l.split("Epoch")[-1])
            for j in range(i + 1, min(i + 4, len(lines))):
                if "," in lines[j]:
                    try:
                        return ep, dict(zip(keys, [float(x) for x in lines[j].split(",")]))
                    except ValueError:
                        pass
    return None, None


# --------------------------------------------------------------------------- #
def dump_one(run_dir, out_root, device, batch_size, dry_run=False):
    from train_utils import generate_model
    from data_util import my_collate_fn
    from main import predict
    from find_threshold import find_threshold_micro
    from evaluation import all_metrics

    tag = os.path.basename(run_dir.rstrip("/"))
    short = tag.split("-")[-1] if "-" in tag else tag
    ckpt = find_checkpoint(run_dir)
    print("\n" + "=" * 74)
    print("cell %s" % short)
    print("=" * 74)
    print("  run directory : %s" % run_dir)
    print("  weights       : %s" % (ckpt or "(none, skipped)"))
    if ckpt is None:
        return None
    out = os.path.join(out_root, short)
    if dry_run:
        print("  [dry-run] would write to %s" % out)
        return None
    os.makedirs(out, exist_ok=True)

    args = load_args(run_dir, device)
    train_ds, dev_ds, test_ds = build_datasets(args)
    print("  dataset       : version=%s | C=%d | train=%d dev=%d test=%d"
          % (args.version, train_ds.code_count, len(train_ds), len(dev_ds), len(test_ds)))

    model = generate_model(args, train_ds)
    obj = torch.load(ckpt, map_location="cpu", weights_only=False)
    if isinstance(obj, torch.nn.Module):
        model = obj
        # The checkpoint is a pickled Module, so the class comes from sys.path.
        #    Print it and assert it lives in the expected repository.
        cls = type(model)
        f = getattr(sys.modules.get(cls.__module__), "__file__", "?")
        print("  model class   : %s.%s  <- %s" % (cls.__module__, cls.__name__, f))
        if not os.path.abspath(f).startswith(REPO):
            raise RuntimeError(
                "the model class comes from %s, not the expected repository %s -- "
                "deserialisation used the wrong one and results would be silently wrong. "
                "Pass --repo to point at the right repository." % (f, REPO))
    else:
        model.load_state_dict(obj)
    model = model.to(device).eval()
    model.c_input_word = train_ds.c_input_word.to(device)
    model.c_word_mask = train_ds.c_word_mask.to(device)

    def mk(ds):
        return DataLoader(ds, batch_size=batch_size, shuffle=False,
                          collate_fn=my_collate_fn)
    dev_dl, test_dl = mk(dev_ds), mk(test_ds)

    print("  [1/4] dev forward and threshold selection ...")
    yhat_d, y_d, prob_d = predict(model, dev_dl, device, None, False, args)
    thr = float(find_threshold_micro(prob_d, y_d))
    pred_d = (prob_d > thr).astype(np.uint8)
    print("        dev threshold = %.4f" % thr)

    print("  [2/4] test forward ...")
    yhat_t, y_t, prob_t = predict(model, test_dl, device, thr, False, args)
    pred_t = (prob_t > thr).astype(np.uint8)

    print("  [3/4] recompute metrics and check against metric_log ...")
    met = all_metrics(yhat=pred_t, y=y_t, yhat_raw=prob_t)
    ep_ref, ref = best_test_from_metric_log(run_dir)
    print("        recomputed best test: " + ", ".join(
        "%s=%.4f" % (k, met[k]) for k in ["auc_macro", "f1_micro", "prec_at_8"]))
    if ref:
        print("        metric_log(%s): " % ("Best_Test_Epoch%d" % ep_ref)
              + ", ".join("%s=%.4f" % (k, ref[k])
                          for k in ["MACRO-auc", "MICRO-f1", "P@8"]))
        d = abs(met["auc_macro"] - ref["MACRO-auc"])
        if d > 2e-3:
            print("        [warn] AUC-macro differs by %.4f -- threshold or checkpoint does "
                  "not match training; the long-tail conclusions would be affected." % d)
        else:
            print("        matches metric_log (|dAUC-macro|=%.4f)" % d)
    else:
        print("        (no Best_Test block yet, skipping the check)")

    print("  [4/4] training frequency and write-out ...")
    freq, n_train_docs, unknown = train_freq(train_ds, train_ds.code_count)
    np.save(os.path.join(out, "y_true.npy"), y_t.astype(np.uint8))
    np.save(os.path.join(out, "y_prob.npy"), prob_t.astype(np.float32))
    np.save(os.path.join(out, "y_pred.npy"), pred_t)
    np.save(os.path.join(out, "dev_y_true.npy"), y_d.astype(np.uint8))
    np.save(os.path.join(out, "dev_y_prob.npy"), prob_d.astype(np.float32))
    np.save(os.path.join(out, "dev_y_pred.npy"), pred_d)
    np.save(os.path.join(out, "train_freq.npy"), freq.astype(np.int32))
    meta = {"cell": short, "run_dir": run_dir, "ckpt": os.path.basename(ckpt),
            "version": args.version, "threshold": thr,
            "n_test": int(len(y_t)), "n_labels": int(train_ds.code_count),
            "n_train_docs": n_train_docs, "labels_unmapped_in_train": int(unknown),
            "metrics_recomputed": {k: float(v) for k, v in met.items()},
            "metric_log_best_test": ref, "metric_log_best_test_epoch": ep_ref,
            "bucket_rule": "train_freq: N=1 / 2-5 / 6-20 / 21-100 / >100"}
    with open(os.path.join(out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    buckets = [("N=1", freq == 1), ("2<=N<=5", (freq >= 2) & (freq <= 5)),
               ("6<=N<=20", (freq >= 6) & (freq <= 20)),
               ("21<=N<=100", (freq >= 21) & (freq <= 100)),
               ("N>100", freq > 100)]
    print("        labels per bucket (training frequency): "
          + ", ".join("%s=%d" % (nm, int(m.sum())) for nm, m in buckets)
          + " | total %d" % int(sum(m.sum() for _, m in buckets)))
    meta["bucket_counts"] = {nm: int(m.sum()) for nm, m in buckets}
    print("        wrote: %s" % out)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--runs", default=os.environ.get("ICD_RUNS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs")))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "longtail"))
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--repo", default=None,
                    help="repository to import the model class from; the checkpoint is a "
                         "pickled Module, so it must be the one used for training. "
                         "Resolution: --repo > ICD_REPO > cwd > main")
    a = ap.parse_args()

    repo, why = resolve_repo(a.repo)
    set_repo(repo, why)
    print("repository: %s (source: %s)" % (REPO, why))

    device = "cuda:0"
    if a.run_dir:
        dirs = [a.run_dir]
    elif a.all:
        dirs = sorted(os.path.join(a.runs, d) for d in os.listdir(a.runs)
                      if os.path.isdir(os.path.join(a.runs, d)))
    else:
        ap.error("give --run-dir or --all")
    metas = []
    for d in dirs:
        try:
            m = dump_one(d, a.out, device, a.batch_size, a.dry_run)
            if m:
                metas.append(m)
        except FileNotFoundError as e:
            print("  skipped (missing file): %s" % e)
        except Exception as e:
            print("  [warn] failed: %r" % e)
    if metas:
        # Merge rather than overwrite: rewriting index.csv from only this run's entries
        # would drop the rows of cells exported earlier. Rows are merged per cell.
        idx_path = os.path.join(a.out, "index.csv")
        head = ["cell", "version", "threshold", "n_test", "n_labels",
                "auc_macro", "f1_micro", "prec_at_8"]
        rows = {}
        if os.path.exists(idx_path):
            with open(idx_path, newline="", encoding="utf-8") as f:
                for r in csv.reader(f):
                    if r and r[0] != "cell" and len(r) == len(head):
                        rows[r[0]] = r
        for m in metas:
            r = m["metrics_recomputed"]
            rows[m["cell"]] = [m["cell"], m["version"], "%.4f" % m["threshold"],
                               m["n_test"], m["n_labels"], "%.4f" % r["auc_macro"],
                               "%.4f" % r["f1_micro"], "%.4f" % r["prec_at_8"]]
        with open(idx_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(head)
            for k in sorted(rows):
                w.writerow(rows[k])
        print("\nsummary: %s (%d rows updated, %d rows in file)" % (idx_path, len(metas), len(rows)))
    print("\nnext step (long-tail buckets): feed the exported cells to bucket_metrics.py:")
    import glob as _g
    cells = sorted(os.path.basename(os.path.dirname(p))
                   for p in _g.glob(os.path.join(a.out, "*", "meta.json")))
    if cells:
        cmd = "%s %s" % (sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "bucket_metrics.py"))
        cmd = "python3 bucket_metrics.py"
        for c in cells:
            d = os.path.join(a.out, c)
            cmd += (" \\\n    --model %s %s/y_true.npy %s/y_prob.npy %s/y_pred.npy"
                    % (c, d, d, d))
        cmd += (" \\\n    --train-freq %s/%s/train_freq.npy --scheme 5buckets"
                % (a.out, cells[0]))
        print(cmd)
    print("  (bucket rules are in the bucket_metrics.py docstring)")

    # A failed check must be visible to the caller, not buried in a log.
    if metas:
        bad = []
        for m in metas:
            ref = m.get("metric_log_best_test")
            if ref:
                d = abs(m["metrics_recomputed"]["auc_macro"] - ref["MACRO-auc"])
                if d > 2e-3:
                    bad.append((m["cell"], d))
        print("\ncheck summary: %d cells, %d disagree with metric_log" % (len(metas), len(bad)))
        for c, d in bad:
            print("  ⚠️ %s  |ΔAUCma|=%.4f" % (c, d))
        if bad:
            sys.exit(1)
        print("  all agree with metric_log")


if __name__ == "__main__":
    main()
