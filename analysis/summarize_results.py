#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""summarize_results.py -- turn the cell outputs into tables ready for the paper.

Input: the run directories of the study, as produced by run_matrix_v2.sh.

Output:
    <out>/tables.md     main table, decisive contrasts and per-epoch curves (Markdown)
    <out>/tables.tex    booktabs table (AUC/F1/P@k, x100)
    <out>/epochs.csv    long form (cell, epoch, split, metric, value), for plotting
    <out>/SUMMARY.txt   console summary

Design notes
------------
* Nothing is guessed: values are read from the run outputs, and a missing cell is marked
  as running or not started.
* ``best`` is whatever dev selected: the Best_*_Epoch{K} block is read, not recomputed.
* Contrasts are paired at the same epoch, which is what isolates one change at a time.
"""
import argparse
import csv
import json
import os
import re

METRICS = ["MACRO-auc", "MICRO-auc", "MACRO-f1", "MICRO-f1",
           "P@5", "P@8", "P@15", "R@5", "R@8", "R@15"]
SHORT = {"MACRO-auc": "AUCma", "MICRO-auc": "AUCmi", "MACRO-f1": "F1ma",
         "MICRO-f1": "F1mi", "P@5": "P@5", "P@8": "P@8", "P@15": "P@15",
         "R@5": "R@5", "R@8": "R@8", "R@15": "R@15"}
#: The six metrics the main table uses (the rest go to the appendix)
MAIN = ["MACRO-auc", "MICRO-auc", "MACRO-f1", "MICRO-f1", "P@8", "P@15"]

CELLS = [
    ("1", "1_msmn_fp32_naive",
     "MSMN (BiLSTM) / naive / FP32"),
    ("2", "2_msmn_bf16_naive",
     "MSMN (BiLSTM) / naive / BF16"),
    ("3", "3_transf_bf16_naive",
     "Transformer (bucketed, non-flash) / naive / BF16"),
    ("4", "4_transf_bf16_memeff",
     "Transformer (bucketed, non-flash) / mem_efficient / BF16"),
    ("5", "5_transf_bf16_varlen_fa",
     "Transformer (varlen + **Flash Attention**) / FA / BF16"),
    ("6", "6_transf_fp32_naive",
     "Transformer (bucketed, non-flash) / naive / FP32"),
]
#: Decisive contrasts: (A, B, description); A->B changes one variable
PAIRS = [
    ("1", "2", "precision, on the baseline (fp32 -> bf16)"),
    ("2", "3", "encoder architecture at fixed precision (BiLSTM -> Transformer)"),
    ("3", "4", "decoder implementation (naive -> mem_efficient)"),
    ("4", "5", "attention kernel (mem_efficient -> flash)"),
    ("3", "6", "precision, on the Transformer (bf16 -> fp32)"),
    ("1", "5", "end to end (cell 1 -> cell 5)"),
]


# --------------------------------------------------------------------------- #
def parse_metric_log(path):
    """→ {'Dev': {1: {...}}, 'Test': {...}, 'Best_Test': (K, {...}), ...}"""
    out = {"Dev": {}, "Test": {}, "Best_Train": None, "Best_Dev": None,
           "Best_Test": None}
    if not os.path.exists(path):
        return out
    lines = [l.rstrip("\n") for l in open(path, encoding="utf-8")]
    i = 0
    while i < len(lines):
        l = lines[i].strip()
        if not l or set(l) <= {"-"}:
            i += 1
            continue
        m = re.match(r"^(Dev|Test)_Epoch(\d+)$", l)
        mb = re.match(r"^Best_(Train|Dev|Test)_Epoch(\d+)$", l)
        if m or mb:
            head = m.group(0) if m else mb.group(0)
            vals = None
            for j in range(i + 1, min(i + 4, len(lines))):
                if re.match(r"^[-\d]", lines[j].strip()) and "," in lines[j]:
                    try:
                        vals = [float(x) for x in lines[j].split(",")]
                    except ValueError:
                        vals = None
                    if vals and len(vals) == len(METRICS):
                        break
            if vals:
                d = dict(zip(METRICS, vals))
                if m:
                    out[m.group(1)][int(m.group(2))] = d
                else:
                    out["Best_" + mb.group(1)] = (int(mb.group(2)), d)
            i += 1
        else:
            i += 1
    return out


def load_epochs(path):
    rows = []
    if os.path.exists(path):
        for l in open(path, encoding="utf-8"):
            l = l.strip()
            if l:
                try:
                    rows.append(json.loads(l))
                except ValueError:
                    pass
    return rows


def cfg_brief(args_json):
    try:
        a = json.load(open(args_json, encoding="utf-8"))
    except Exception:
        return "?"
    bits = [a.get("combiner", "?"), "bf16" if a.get("bf16") else "fp32"]
    if a.get("label_repr"):
        bits.append("label=" + str(a["label_repr"]))
    if a.get("attn_impl"):
        bits.append("dec=" + str(a["attn_impl"]))
    if a.get("dec_chunk_codes"):
        bits.append("dck" + str(a["dec_chunk_codes"]))
    return " ".join(bits)


def find_run(runs_dir, tag):
    """Run directory names end with -<tag>, which generate_output_folder_name appends."""
    if not os.path.isdir(runs_dir):
        return None
    hits = [d for d in os.listdir(runs_dir)
            if d.endswith(tag) or d.endswith("-" + tag)]
    if not hits:
        return None
    # take the most recently modified
    hits.sort(key=lambda d: os.path.getmtime(os.path.join(runs_dir, d)))
    return os.path.join(runs_dir, hits[-1])


# --------------------------------------------------------------------------- #
def collect(runs_dir):
    data = {}
    for cid, tag, desc in CELLS:
        rd = find_run(runs_dir, tag)
        if rd is None:
            data[cid] = {"status": "not started", "desc": desc, "tag": tag}
            continue
        ml = parse_metric_log(os.path.join(rd, "metric_log"))
        ep = load_epochs(os.path.join(rd, "epochs.jsonl"))
        best = ml.get("Best_Test")
        data[cid] = {
            "status": "done" if best else ("running" if ep else "not started"),
            "desc": desc, "tag": tag, "dir": rd, "metric_log": ml,
            "epochs": ep, "cfg": cfg_brief(os.path.join(rd, "args.json")),
            "best_test": best[1] if best else None,
            "best_dev": (ml.get("Best_Dev") or (None, None))[1],
            "best_train": (ml.get("Best_Train") or (None, None))[1],
            "best_epoch": best[0] if best else None,
            "n_epochs": len(ep),
            "h_per_epoch": (sum(r.get("epoch_seconds", 0) for r in ep) / len(ep) / 3600
                            if ep else None),
            "peak_gb": max((r.get("peak_mem_GB", 0) for r in ep), default=None),
            "step_ms": (sum(r.get("step_ms_mean", 0) for r in ep) / len(ep)
                        if ep else None),
        }
    return data


def f100(v, nd=1):
    return "-" if v is None else ("%.*f" % (nd, v * 100))


def pct(d, k):
    return f100(None if d is None else d.get(k))


def table_main_md(D):
    L = ["| cell | configuration | " + " | ".join(SHORT[m] for m in MAIN)
         + " | h/epoch | peak | status |",
         "|---|---|" + "---:|" * (len(MAIN) + 3)]
    for cid, tag, desc in CELLS:
        d = D[cid]
        bt = d.get("best_test")
        cells = " | ".join(pct(bt, m) for m in MAIN)
        hp = "-" if d.get("h_per_epoch") is None else "%.2f" % d["h_per_epoch"]
        pk = "-" if d.get("peak_gb") is None else "%.2f GB" % d["peak_gb"]
        L.append("| %s | %s | %s | %s | %s | %s |"
                 % (cid, desc, cells, hp, pk, d["status"]))
    return "\n".join(L)


def table_main_tex(D):
    L = [r"\begin{tabular}{ll" + "r" * len(MAIN) + "}",
         r"\toprule",
         "Grid & Configuration & " + " & ".join(SHORT[m] for m in MAIN) + r" \\",
         r"\midrule"]
    for cid, tag, desc in CELLS:
        d = D[cid]
        bt = d.get("best_test")
        cells = " & ".join(pct(bt, m) for m in MAIN)
        L.append("%s & %s & %s \\\\" % (cid, desc.replace("**", ""), cells))
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def table_pairs_md(D):
    L = ["| contrast | variable | metric | A | B | d(B-A) | shared epochs |", "|---|---|---|---:|---:|---:|---:|"]
    for a, b, what in PAIRS:
        ea, eb = D[a].get("epochs") or [], D[b].get("epochs") or []
        na = {r["epoch"]: (r.get("test") or {}) for r in ea}
        nb = {r["epoch"]: (r.get("test") or {}) for r in eb}
        common = sorted(set(na) & set(nb))
        if not common:
            L.append("| %s→%s | %s | — | - | - | - | 0 |" % (a, b, what))
            continue
        k = common[-1]
        for m in ("auc_macro", "f1_micro", "prec_at_8"):
            va, vb = na[k].get(m), nb[k].get(m)
            if va is None or vb is None:
                continue
            L.append("| %s→%s | %s | %s | %.4f | %.4f | **%+.4f** | %d (ep%d) |"
                     % (a, b, what, m, va, vb, vb - va, len(common), k))
    return "\n".join(L)


def curves_md(D):
    out = []
    for cid, tag, desc in CELLS:
        ep = D[cid].get("epochs") or []
        if not ep:
            continue
        out.append("\n**cell %s** -- %s\n" % (cid, desc))
        out.append("| ep | train loss | dev AUCma | test AUCma | test F1mi | test P@8 | test P@15 |")
        out.append("|---:|---:|---:|---:|---:|---:|---:|")
        for r in ep:
            t = r.get("test") or {}
            dv = r.get("dev") or {}
            out.append("| %d | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f |"
                       % (r["epoch"], r.get("train_loss", 0),
                          dv.get("auc_macro", 0), t.get("auc_macro", 0),
                          t.get("f1_micro", 0), t.get("prec_at_8", 0),
                          t.get("prec_at_15", 0)))
    return "\n".join(out)


def dump_csv(D, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cell", "epoch", "split", "metric", "value"])
        for cid, tag, desc in CELLS:
            for r in D[cid].get("epochs") or []:
                for split in ("dev", "test"):
                    for m, v in (r.get(split) or {}).items():
                        w.writerow([cid, r["epoch"], split, m, v])


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="experiments/formal_results/experiments/runs_v2")
    ap.add_argument("--out", default="experiments/formal_results/summary")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    D = collect(a.runs)
    md = ["# Cell summary (generated by `summarize_results.py`)", "",
          "> Every number is read from the run directories' metric_log / epochs.jsonl; none is typed by hand.",
          "> `best` is what dev selected (the Best_*_Epoch{K} block). Metrics are x100.", "",
          "## Table A: main table (best test)", "", table_main_md(D), "",
          "## Table B: decisive contrasts (paired at the same epoch)", "",
          table_pairs_md(D), "",
          "## Table C: per-epoch curves", "", curves_md(D), ""]
    open(os.path.join(a.out, "tables.md"), "w", encoding="utf-8").write("\n".join(md))
    open(os.path.join(a.out, "tables.tex"), "w", encoding="utf-8").write(
        "% Table A: best test (x100)\n" + table_main_tex(D) + "\n")
    dump_csv(D, os.path.join(a.out, "epochs.csv"))

    txt = ["Table A: main (best test, x100)", table_main_md(D), "",
           "Table B: decisive contrasts", table_pairs_md(D), "",
           "cell status:"]
    for cid, tag, desc in CELLS:
        d = D[cid]
        txt.append("  %s %-52s %-6s %s/20 epochs%s"
                   % (cid, desc, d["status"], d.get("n_epochs", 0),
                      "" if d.get("h_per_epoch") is None
                      else "  %.2f h/epoch  peak %.2f GB" % (d["h_per_epoch"], d["peak_gb"])))
    out = "\n".join(txt)
    open(os.path.join(a.out, "SUMMARY.txt"), "w", encoding="utf-8").write(out)
    print(out)
    print("\nwrote: %s/{tables.md,tables.tex,epochs.csv,SUMMARY.txt}" % a.out)


if __name__ == "__main__":
    main()
