#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
bucket_metrics.py -- compute ICD coding metrics per training-frequency bucket
(the long-tail analysis of the paper).

It does three things:
  1. counts, per ICD code, how many training documents contain it (N), and buckets labels by N;
  2. for each model, computes macro-F1 / micro-F1 / macro-AUC / micro-AUC within each bucket;
  3. prints rows that paste straight into a LaTeX table, for both a 5-bucket and a 3-bucket
     scheme.

Input format
------------
Each model needs three arrays:
  y_true : (n_test_docs, n_labels) 0/1 matrix, .npy or .npz
  y_prob : (n_test_docs, n_labels) probability matrix, .npy or .npz
  y_pred : (n_test_docs, n_labels) 0/1 prediction matrix, .npy or .npz

Training frequencies come either way:
  --train-freq   train_freq.npy     integer array (n_labels,), entry i = documents containing label i
  --train-labels train_labels.npy   0/1 matrix (n_train_docs, n_labels), summed here

The column order of y_true and y_pred must match train_freq: the same label in the same
column.
"""

import argparse
import sys

import numpy as np


# ----------------------------------------------------------------------------
# Bucket definitions. Change them here only; the rest of the script is independent.
# ----------------------------------------------------------------------------
BUCKET_SCHEMES = {
    "5buckets": {
        "name": "5-frequency buckets (Vu et al., 2020)",
        "edges": [(1, 1), (2, 5), (6, 20), (21, 100), (101, np.inf)],
        "labels": ["$N=1$", "$2 \\le N \\le 5$", "$6 \\le N \\le 20$",
                   "$21 \\le N \\le 100$", "$N > 100$"],
    },
    "3buckets": {
        "name": "3-frequency buckets (<10 / 10-50 / >50)",
        "edges": [(1, 9), (10, 50), (51, np.inf)],
        "labels": ["$N < 10$", "$10 \\le N \\le 50$", "$N > 50$"],
    },
}


# ----------------------------------------------------------------------------
# Metrics implemented without sklearn, to avoid version drift; behaviour matches sklearn.
# ----------------------------------------------------------------------------
def f1_macro_micro(y_true, y_pred):
    """Return (macro_f1, micro_f1) in percent. Columns are labels."""
    tp = (y_true * y_pred).sum(axis=0).astype(float)
    fp = ((1 - y_true) * y_pred).sum(axis=0).astype(float)
    fn = (y_true * (1 - y_pred)).sum(axis=0).astype(float)

    denom = 2 * tp + fp + fn
    per_label_f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)

    TP, FP, FN = tp.sum(), fp.sum(), fn.sum()
    micro = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0.0
    return 100.0 * per_label_f1.mean(), 100.0 * micro


def auc_macro_micro(y_true, y_prob):
    """
    Return (macro_auc, micro_auc) in percent.

    Implementation notes, a common source of wrong numbers in papers:
      - macro AUC averages per-label AUCs. A label with a single class in the test split
        cannot be scored and is skipped by convention, so the denominator is the number of
        scorable labels, not all labels in the bucket.
      - micro AUC flattens every (document, label) score into one global AUC, which is
        dominated by frequent labels and therefore moves very little across buckets.
      - AUC is computed from the Mann-Whitney U statistic with tie-corrected ranks,
        O(n log n), without sklearn.
    """
    # ---- macro AUC, per label ----
    per_label = []
    for j in range(y_true.shape[1]):
        t = y_true[:, j]
        s = y_prob[:, j]
        n_pos = float(t.sum())
        n_neg = float(len(t) - n_pos)
        if n_pos == 0 or n_neg == 0:
            continue                      # single-class label in this split, skipped
        per_label.append(_auc(s, t))
    macro = 100.0 * float(np.mean(per_label)) if per_label else float("nan")

    # ---- micro AUC, flattened ----
    micro = 100.0 * _auc(y_prob.ravel(), y_true.ravel())
    return macro, micro


def _auc(scores, labels):
    """Standard AUC with tie handling, equivalent to sklearn.metrics.roc_auc_score."""
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    sorted_scores = scores[order]

    # tied scores share the average rank
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0    # ranks start at 1
        i = j + 1

    n_pos = float(labels.sum())
    n_neg = float(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


# ----------------------------------------------------------------------------
# Loading and validation
# ----------------------------------------------------------------------------
def load_matrix(path):
    if path.endswith(".npz"):
        data = np.load(path)
        key = list(data.keys())[0]
        return data[key]
    return np.load(path)


def load_train_freq(args):
    if args.train_freq:
        freq = np.load(args.train_freq).astype(int).ravel()
    elif args.train_labels:
        tl = load_matrix(args.train_labels)
        freq = tl.sum(axis=0).astype(int).ravel()
    else:
        sys.exit("give either --train-freq or --train-labels (see the module docstring)")
    return freq


def build_buckets(freq, scheme):
    edges = BUCKET_SCHEMES[scheme]["edges"]
    masks = []
    for lo, hi in edges:
        masks.append((freq >= lo) & (freq <= hi))
    return masks


# ----------------------------------------------------------------------------
# Main flow
# ----------------------------------------------------------------------------
def compute(model_name, y_true, y_prob, y_pred, freq, scheme):
    masks = build_buckets(freq, scheme)
    result = {"method": model_name, "buckets": [], "all": None}

    for m in masks:
        if m.sum() == 0:
            result["buckets"].append(dict(n=int(m.sum()), f1m=np.nan, aucm=np.nan,
                                          f1mi=np.nan, aucmi=np.nan))
            continue
        f1m, f1mi = f1_macro_micro(y_true[:, m], y_pred[:, m])
        aucm, aucmi = auc_macro_micro(y_true[:, m], y_prob[:, m])
        result["buckets"].append(dict(n=int(m.sum()), f1m=f1m, aucm=aucm,
                                      f1mi=f1mi, aucmi=aucmi))

    f1m, f1mi = f1_macro_micro(y_true, y_pred)
    aucm, aucmi = auc_macro_micro(y_true, y_prob)
    result["all"] = dict(n=int(y_true.shape[1]), f1m=f1m, aucm=aucm, f1mi=f1mi, aucmi=aucmi)
    return result


def fmt(x, width=5):
    return "  --  " if (x is None or (isinstance(x, float) and np.isnan(x))) else ("%*.1f" % (width, x))


def print_table(results, scheme, metric="f1m"):
    meta = BUCKET_SCHEMES[scheme]
    title = {"f1m": "Macro-F1", "aucm": "Macro-AUC",
             "f1mi": "Micro-F1", "aucmi": "Micro-AUC"}[metric]
    print("\n=== %s --- %s ===" % (meta["name"], title))
    header = "%-12s" % "Method" + "".join("| %-9s" % meta["labels"][i] for i in range(len(meta["labels"]))) + "| %-8s" % "All"
    print(header)
    print("-" * len(header))
    for r in results:
        row = "%-12s" % r["method"]
        for b in r["buckets"]:
            row += "| %-9s" % fmt(b[metric])
        row += "| %-8s" % fmt(r["all"][metric])
        print(row)
    row = "%-12s" % "#Codes"
    for b in results[0]["buckets"]:
        row += "| %-9d" % b["n"]
    row += "| %-8d" % results[0]["all"]["n"]
    print(row)


def latex_rows(results, scheme):
    """Emit table rows ready to paste: macro-F1 and macro-AUC alternating, as the paper does."""
    lines = []
    for r in results:
        name = "\\textbf{%s}" % r["method"] if r["method"].lower().startswith("fa") else r["method"]
        cells = []
        for b in r["buckets"]:
            cells.append(fmt(b["f1m"]).strip())
            cells.append(fmt(b["aucm"]).strip())
        cells.append(fmt(r["all"]["f1m"]).strip())
        body = " & ".join(cells)
        lines.append("%-28s & %s \\\\" % (name, body))
    counts = " & ".join(str(b["n"]) for b in results[0]["buckets"]) + " & %d" % results[0]["all"]["n"]
    # Two columns per bucket (F1, AUC); the #Codes row fills the F1 column only
    spaced = " & ".join("%d &" % b["n"] for b in results[0]["buckets"]) + " %d" % results[0]["all"]["n"]
    lines.append("\\midrule")
    lines.append("%-28s & %s \\\\" % ("\\# Codes", spaced))
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", nargs=4, action="append", metavar=("NAME", "TRUE", "PROB", "PRED"),
                    required=True, help="model name plus y_true/y_prob/y_pred paths; may be repeated")
    ap.add_argument("--train-freq", help="training frequency array, shape (n_labels,)")
    ap.add_argument("--train-labels", help="training 0/1 matrix, shape (n_train, n_labels)")
    ap.add_argument("--scheme", choices=list(BUCKET_SCHEMES), default="5buckets")
    ap.add_argument("--out", default="latex_rows.tex", help="output path for the LaTeX rows")
    args = ap.parse_args()

    freq = load_train_freq(args)
    print("labels: %d; frequency range: %d ~ %d" % (len(freq), freq.min(), freq.max()))

    results = []
    for name, true_p, prob_p, pred_p in args.model:
        y_true = load_matrix(true_p)
        y_prob = load_matrix(prob_p)
        y_pred = load_matrix(pred_p)
        if not (y_true.shape == y_prob.shape == y_pred.shape):
            sys.exit("shape mismatch: %s" % name)
        if y_true.shape[1] != len(freq):
            sys.exit("columns (%d) differ from train_freq length (%d); check the label column order"
                     % (y_true.shape[1], len(freq)))
        print("loaded %-14s y_true%s  positive rate %.4f" % (name, y_true.shape, y_true.mean()))
        results.append(compute(name, y_true, y_prob, y_pred, freq, args.scheme))

    for metric in ("f1m", "aucm", "f1mi", "aucmi"):
        print_table(results, args.scheme, metric)

    rows = latex_rows(results, args.scheme)
    with open(args.out, "w") as fh:
        fh.write("\n".join(rows) + "\n")
    print("\nLaTeX rows written to %s (macro-F1 / macro-AUC alternating)" % args.out)

    print("""
Notes:
  1. Bucket by TRAINING frequency, not test frequency, or the N=1 bucket ends up almost empty.
  2. Macro AUC skips labels with a single class in the test split. If many such labels exist,
     state in a footnote how many labels actually entered the average.
  3. Micro metrics barely move across buckets because frequent labels dominate them.
""")


if __name__ == "__main__":
    main()
