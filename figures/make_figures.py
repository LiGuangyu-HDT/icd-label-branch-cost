#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""make_figures.py -- generate Fig. 1-5 of the paper (reads local run outputs, no GPU).

Input: <results root>/{runs_v2,runs_seeds,runs_diag,runs_probes}/*/epochs.jsonl
Epoch selection follows the protocol: the epoch with the highest dev P@8, reporting that
epoch's test metrics.

The results root is resolved, in order, from:
  1. --root on the command line,
  2. $ICD_RESULTS_ROOT,
  3. <umbrella>/experiments/formal_results/experiments (the layout the paper was made in),
  4. the repository itself (what study/run_matrix_v2.sh produces with its default OUT_DIR).
Nothing here is tied to one machine: any of the four reaches the same code path.

Fig. 5 reads analysis/bucket_metrics.py's JSON output when it is present, and says so on
stdout when it has to fall back to the values the published figure was drawn from.

Output (PDF for LaTeX, PNG for preview): <out>/fig1..fig5.{pdf,png}, default figures/out/.

Usage:  python3 make_figures.py [--root PATH] [--out PATH] [--bucket-json PATH]
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))            # repository root
ROOT = os.path.abspath(os.path.join(REPO, ".."))            # the author's umbrella directory


def _root_candidates():
    """Places the run directories may live, most specific first."""
    out = []
    env = os.environ.get("ICD_RESULTS_ROOT")
    if env:
        out.append(env)
    out.append(os.path.join(ROOT, "experiments", "formal_results", "experiments"))
    out.append(REPO)
    return out


def resolve_results_root(explicit=None):
    if explicit:
        return os.path.abspath(explicit)
    for cand in _root_candidates():
        if os.path.isdir(os.path.join(cand, "runs_v2")):
            return cand
    return _root_candidates()[0]


MIRROR = resolve_results_root()
OUT = os.environ.get("ICD_FIGURES_OUT") or os.path.join(REPO, "figures", "out")
#: JSON written by analysis/bucket_metrics.py --json-out, the source of Fig. 5.
BUCKET_JSON = os.environ.get("ICD_BUCKET_JSON") or os.path.join(MIRROR, "longtail",
                                                               "bucket_metrics.json")

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "legend.fontsize": 7, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.bbox": "tight",
})

CELLS = ["1_msmn_fp32_naive", "2_msmn_bf16_naive", "3_transf_bf16_naive",
         "5_transf_bf16_varlen_fa", "6_transf_fp32_naive",
         "7_hybrid_lstm_label_fa_text", "8_hybrid_labelfp32"]
NICE = {"1_msmn_fp32_naive": "cell 1: BiLSTM fp32", "2_msmn_bf16_naive": "cell 2: BiLSTM bf16",
        "3_transf_bf16_naive": "cell 3: Transf. bf16\n(non-flash)", "5_transf_bf16_varlen_fa": "cell 5: Transf. bf16\n(flash)",
        "6_transf_fp32_naive": "cell 6: Transf. fp32", "7_hybrid_lstm_label_fa_text": "cell 7: hybrid\n(label bf16)",
        "8_hybrid_labelfp32": "cell 8: hybrid\n(label fp32)",
        # cell 4 is a decoder-only micro-benchmark: it is not on the published figures, but
        # it is in the long-tail dump, so Fig. 5 must be able to label it rather than skip it
        "4_transf_bf16_memeff": "cell 4: Transf. bf16\n(mem-efficient)"}
COL = {"1_msmn_fp32_naive": "#1b5e20", "2_msmn_bf16_naive": "#c62828",
       "3_transf_bf16_naive": "#1565c0", "5_transf_bf16_varlen_fa": "#00838f",
       "6_transf_fp32_naive": "#37474f", "7_hybrid_lstm_label_fa_text": "#8e24aa",
       "8_hybrid_labelfp32": "#e65100", "4_transf_bf16_memeff": "#6d4c41"}


def recs(sub, tag):
    d = [x for x in glob.glob(os.path.join(MIRROR, sub, "*")) if x.endswith(tag)]
    if not d:
        return None
    return [json.loads(l) for l in open(os.path.join(d[0], "epochs.jsonl"), encoding="utf-8")]


def best(sub, tag):
    """Select the epoch on dev P@8, as the protocol does, and return that record."""
    rs = recs(sub, tag)
    if not rs:
        return None
    return max(rs, key=lambda r: r["dev"]["prec_at_8"])


def full(tag):
    return best("runs_v2", tag)


# Full-scale cells with extra seeds: 1 and 2. The others remain single runs.
FULL_REPLICATED = ("1_msmn_fp32_naive", "2_msmn_bf16_naive")


def full_seed_stats(tag):
    """Cells 1/2 at full scale: seed 1 from runs/ plus s4/s5, summarised as mean +- s.d.
    Returns ({metric: (mean, sd)}, n); with n<2 the s.d. is 0, so the error bar is 0."""
    TOTAL_EPOCHS = 20          # the full-scale protocol fixes 20 epochs; unfinished runs are excluded
    runs = []
    b = best("runs_v2", tag)
    if b:
        runs.append(b)
    for lab in ("s4", "s5"):
        b = best("runs_seeds", "%s_%s" % (tag, lab))
        if b and len(recs("runs_seeds", "%s_%s" % (tag, lab)) or []) >= TOTAL_EPOCHS:
            runs.append(b)
    out = {}
    for k in ("auc_macro", "f1_micro", "prec_at_8", "f1_macro"):
        v = [(r.get("test") or {}).get(k) for r in runs]
        v = [x for x in v if x is not None]
        if not v:
            continue
        m = float(np.mean(v))
        sd = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
        out[k] = (m, sd)
    return out, len(runs)


def seed_stats(tag):
    """50-code, three seeds: select each seed on dev P@8 and return (mean, std, raw)."""
    vals = {}
    for s in (1, 2, 3):
        b = best("runs_seeds", "%s_s%d" % (tag, s))
        if b:
            for k in ("auc_macro", "f1_micro", "prec_at_8", "f1_macro"):
                vals.setdefault(k, []).append(b["test"][k])
    return ({k: (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0) for k, v in vals.items()}, vals)


# ─────────────────────────────────────────────────────────────────────────────
# Fig. 1: structure and where the cost wall sits
# ─────────────────────────────────────────────────────────────────────────────
def fig1():
    """Conceptual diagram: where the cost sits in a fusion ICD-coding model."""
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.set_xlim(0, 100); ax.set_ylim(0, 58); ax.axis("off")

    BLUE, BLUE_E = "#dce9f5", "#5b7fa6"
    ORAN, ORAN_E = "#fbeee0", "#c08a52"
    GREY, GREY_E = "#eef1f3", "#6b7785"
    INK, ACC, MUT = "#1f2429", "#a8432c", "#767f88"

    def box(x, y, w, h, text, fc, ec, fs=7.1, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.0,rounding_size=1.6",
                                    fc=fc, ec=ec, lw=0.8, zorder=2))
        ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs,
                color=INK, linespacing=1.35, zorder=3,
                fontweight=("bold" if bold else "normal"))

    def arr(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=7, color=GREY_E, lw=0.9,
                                     shrinkA=1, shrinkB=1, zorder=1))

    # ---- inputs ----
    box(1, 37, 21, 11, "clinical note\n$B\\times4{,}000$ tokens", BLUE, BLUE_E)
    box(1, 8, 21, 15, "code descriptions\n$C\\times M=8{,}921\\times4$\nslots", ORAN, ORAN_E)

    # ---- encoders ----
    box(27, 36, 25, 13, "text encoder\nBiLSTM $2\\times$256\nor Transformer $4\\times$512", BLUE, BLUE_E)
    box(27, 6, 25, 19, "label-description\nfusion branch\n(encoder over\ndescriptions)", ORAN, ORAN_E, bold=False)

    # ---- fusion and decoding ----
    box(57.5, 27, 17, 12, "fusion\n$\\rightarrow$ code\nrepresentations", GREY, GREY_E)
    box(79, 27, 20, 12, "decoder\n8,921-way\n(code-axis chunking)", GREY, GREY_E)

    # ---- arrows ----
    arr(22.4, 42.5, 26.6, 42.5); arr(22.4, 15.5, 26.6, 15.5)
    arr(52.4, 42.5, 57.6, 35.0); arr(52.4, 15.5, 57.6, 31.0)
    arr(74.9, 33.0, 78.6, 33.0)

    # ---- annotations in the empty band ----
    ax.text(50, 55.0,
            "every slot is recomputed on every device at every step  ($\\Rightarrow$ trained branch "
            "takes part in every backward pass)",
            ha="center", va="center", fontsize=6.6, color=MUT, style="italic")
    ax.text(50, 1.8,
            "1,070,520 padded slots for 200,353 real tokens = 18.7%  ($5.34\\times$ waste; padding is not masked out)",
            ha="center", va="center", fontsize=6.6, color=MUT, style="italic")
    # the two key quantities on the right, emphasised
    ax.text(89, 46.5, "94 KB per real token\n$\\Rightarrow$ 18.8 GB predicted, 18.48 GB measured",
            ha="center", va="center", fontsize=6.6, color=ACC, linespacing=1.4)
    ax.text(89, 20.0, "fits one 24 GB GPU\nin fp32 and bf16",
            ha="center", va="center", fontsize=6.6, color=ACC, linespacing=1.4)

    fig.savefig(os.path.join(OUT, "fig1.pdf"), bbox_inches="tight", pad_inches=0.08)
    fig.savefig(os.path.join(OUT, "fig1.png"), bbox_inches="tight", pad_inches=0.08, dpi=300)
    plt.close(fig)




def steady_hours(tag):
    """Median epoch_seconds across epochs, in hours, excluding contaminated stretches."""
    import statistics as _st
    d = None
    for base in ("runs_v2", "runs_seeds"):
        for cand in glob.glob(os.path.join(MIRROR, base, "*")):
            nm = os.path.basename(cand)
            if nm.startswith("mimic3_") and nm.endswith(tag):
                d = cand
                break
        if d:
            break
    if not d:
        b = full(tag)
        return b["epoch_seconds"] / 3600.0
    rows = [json.loads(l) for l in open(os.path.join(d, "epochs.jsonl"), encoding="utf-8") if l.strip()]
    return _st.median([r["epoch_seconds"] for r in rows]) / 3600.0


def fig2():
    rows = []
    for t in CELLS:
        b = full(t)
        if not b:
            continue
        # Use the median over epochs: cell 8's epochs 17-20 were contaminated by a
        # co-scheduled job (775 ms against a steady 426 ms), and its selected epoch falls
        # inside that window, so raw epoch_seconds cannot be used.
        hrs = steady_hours(t)
        rows.append((t, b["peak_mem_GB"], hrs, b["test"]["auc_macro"] * 100))
    fig, ax = plt.subplots(figsize=(4.6, 2.75))
    off = {"1_msmn_fp32_naive": (-6, -22), "2_msmn_bf16_naive": (0, 12),
           "3_transf_bf16_naive": (0, 13), "5_transf_bf16_varlen_fa": (34, 12),
           "6_transf_fp32_naive": (0, -20), "7_hybrid_lstm_label_fa_text": (10, 14),
           "8_hybrid_labelfp32": (10, 20)}
    for t, mem, h, auc in rows:
        ax.scatter(mem, h, s=(auc - 45) * 9, color=COL[t], alpha=0.85, zorder=3,
                   edgecolor="k", linewidth=0.4)
        _ha = "center"
        ax.annotate(NICE[t], (mem, h), textcoords="offset points", xytext=off[t],
                    ha=_ha, fontsize=6.6, color=COL[t], zorder=5,
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.88))
    # Non-dominated frontier in (memory, time), healthy cells only (cell 7 collapsed)
    pts = sorted([(m, h) for t, m, h, _ in rows if t != "7_hybrid_lstm_label_fa_text"])
    front, best_h = [], 1e9
    for m, h in pts:
        if h < best_h:
            front.append((m, h)); best_h = h
    ax.plot([p[0] for p in front], [p[1] for p in front], "k--", lw=0.9, alpha=0.7, zorder=1,
            label="Pareto (memory vs time), healthy cells")
    ax.set_ylim(0, 4.1)
    ax.set_xlabel("peak memory per step (GB)")
    ax.set_ylabel("hours per epoch")
    ax.set_title("Cost–accuracy landscape (marker area $\\propto$ AUC-macro)")
    ax.legend(loc="upper right", frameon=False)
    fig.savefig(os.path.join(OUT, "fig2.pdf")); fig.savefig(os.path.join(OUT, "fig2.png"))
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Fig. 3: precision x architecture (full scale and 50-code)
# ─────────────────────────────────────────────────────────────────────────────
def fig3():
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 3.35))
    groups = ["BiLSTM", "Transformer"]
    keys = {"BiLSTM": ("1_msmn_fp32_naive", "2_msmn_bf16_naive"),
            "Transformer": ("6_transf_fp32_naive", "3_transf_bf16_naive")}
    for j, (metric, ylab) in enumerate([("auc_macro", "test AUC-macro"),
                                        ("f1_micro", "test micro-F1")]):
        ax = axes[0][j]
        w = 0.36
        for i, g in enumerate(groups):
            for k, (tag, lab, col) in enumerate([(keys[g][0], "fp32", "#1b5e20"),
                                                 (keys[g][1], "bf16", "#c62828")]):
                if tag in FULL_REPLICATED:
                    st_full, _n = full_seed_stats(tag)
                    if metric not in st_full:      # cell not run yet: leave the slot empty
                        continue
                    v, sd = st_full[metric][0] * 100, st_full[metric][1] * 100
                    ax.bar(i + (k - 0.5) * w, v, w, yerr=sd, color=col, alpha=0.9,
                           error_kw=dict(lw=0.8, capsize=2),
                           label=lab if i == 0 else None)
                else:
                    b = full(tag)
                    if not b:
                        continue
                    v = b["test"][metric] * 100
                    ax.bar(i + (k - 0.5) * w, v, w, color=col, alpha=0.9,
                           label=lab if i == 0 else None)
                _sd = sd if tag in FULL_REPLICATED else 0.0
                ax.text(i + (k - 0.5) * w, v + _sd + (0.5 if metric == "auc_macro" else 1.0),
                        "%.2f" % v, ha="center", fontsize=6.2)
        ax.set_xticks(range(len(groups))); ax.set_xticklabels(groups)
        ax.set_ylabel(ylab)
        if metric == "auc_macro":
            ax.set_ylim(80, 107); ax.set_xlim(-0.75, 1.75)
            ax.set_title("full MIMIC-III (mean ± s.d. where replicated)", fontsize=7.5)
            ax.legend(frameon=False, loc="upper left", ncol=2, framealpha=0.95)
            ax.annotate("", xy=(-0.42, 95.24), xytext=(-0.42, 84.79),
                        arrowprops=dict(arrowstyle="<->", color="#c62828", lw=1.0))
            ax.text(-0.47, 90.0, "$-10.44$", fontsize=6.6, color="#c62828", ha="right", va="center")
            ax.text(1.0, 96.6, "$+0.35$", fontsize=6.6, color="#c62828", ha="center")
        else:
            ax.set_ylim(40, 64); ax.set_title(" ")

        # 50-code setting
        ax = axes[1][j]
        for i, g in enumerate(groups):
            for k, (tag, lab, col) in enumerate([(keys[g][0], "fp32", "#1e88e5"),
                                                 (keys[g][1], "bf16", "#f9a825")]):
                st, raw = seed_stats(tag)
                if metric not in st:
                    # no fp32 Transformer was run in the 50-code setting; mark n/a rather than leave a gap
                    ax.text(i + (k - 0.5) * w, 3, "n/a", ha="center", fontsize=6.0, color="#90a4ae")
                    continue
                m, s = st[metric]
                v = m * 100
                ax.bar(i + (k - 0.5) * w, v, w, yerr=s * 100, color=col, alpha=0.9,
                       error_kw=dict(lw=0.8, capsize=2), label=lab if i == 0 else None)
                ax.text(i + (k - 0.5) * w, v + s * 100 + 2, "%.2f" % v, ha="center", fontsize=6.2)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([g + "\n(cells 3, 5\nunconverged)" if g == "Transformer" else g for g in groups], fontsize=6.6)
        ax.set_ylabel(ylab); ax.set_ylim(0, 100 if metric == "auc_macro" else 80)
        if metric == "auc_macro":
            ax.set_title("MIMIC-III-50 (3 seeds, mean$\\pm$s.d.)")
            ax.legend(frameon=False, loc="center right", bbox_to_anchor=(0.99, 0.30))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3.pdf")); fig.savefig(os.path.join(OUT, "fig3.png"))
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Fig. 4: label scale modulates precision sensitivity
# ─────────────────────────────────────────────────────────────────────────────
def fig4():
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.9))
    # (a) the cell 1 - cell 2 effect against label-space size
    ax = axes[0]
    st50, _ = seed_stats("1_msmn_fp32_naive"); st50b, _ = seed_stats("2_msmn_bf16_naive")
    s1, n1 = full_seed_stats("1_msmn_fp32_naive"); s2, n2 = full_seed_stats("2_msmn_bf16_naive")
    have = all("auc_macro" in d for d in (st50, st50b, s1, s2))
    if have:
        d50 = (st50["auc_macro"][0] - st50b["auc_macro"][0]) * 100
        e50 = np.sqrt(st50["auc_macro"][1] ** 2 + st50b["auc_macro"][1] ** 2) * 100
        # Full scale: difference of three-run means with the combined s.e. of independent samples
        dfull = (s1["auc_macro"][0] - s2["auc_macro"][0]) * 100
        efull = np.sqrt(s1["auc_macro"][1] ** 2 / max(n1, 1)
                        + s2["auc_macro"][1] ** 2 / max(n2, 1)) * 100
        ax.bar([0, 1], [dfull, d50], yerr=[efull, e50], color=["#c62828", "#90a4ae"],
               error_kw=dict(lw=0.8, capsize=3))
        for x, v, e in zip([0, 1], [dfull, d50], [efull, e50]):
            ax.text(x, v + e + 0.25, "%.2f" % v, ha="center", fontsize=7)
        ax.text(1, d50 + 2.2, "within seed noise\n($\\pm$0.1)", ha="center", fontsize=6.4,
                color="#546e7a")
    else:
        ax.text(0.5, 6.6, "cells 1/2 (full and 50-code) not found\nin the results root",
                ha="center", fontsize=7, color="#90a4ae")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["full\n35,684 slots", "MIMIC-III-50\n200 slots"])
    ax.set_ylabel("precision cost $\\Delta$AUC-macro (pts)")
    ax.set_title("(a) same pair, two label-space scales")
    ax.set_ylim(0, 13.2)
    # (b) label-side precision in the hybrid cells
    ax = axes[1]
    r7, r8 = full("7_hybrid_lstm_label_fa_text"), full("8_hybrid_labelfp32")
    h50 = best("runs_diag", "7b_hybrid_50code_diag")

    def _auc(rec):
        return np.nan if not rec else rec["test"]["auc_macro"] * 100

    vals = [_auc(h50), _auc(r7), _auc(r8)]
    labs = ["hybrid, label bf16\n(50 codes,\n3 epochs)", "hybrid, label bf16\n(full)", "hybrid, label fp32\n(full)"]
    cols = ["#a5d6a7", "#8e24aa", "#e65100"]
    ax.bar(range(3), [0 if np.isnan(v) else v for v in vals], color=cols)
    for x, v in enumerate(vals):
        if np.isnan(v):
            ax.text(x, 4, "not run", ha="center", fontsize=6.0, color="#90a4ae")
        else:
            ax.text(x, v + 2, "%.2f" % v, ha="center", fontsize=7)
    ax.set_xticks(range(3)); ax.set_xticklabels(labs, fontsize=6.4)
    ax.set_ylabel("test AUC-macro"); ax.set_ylim(0, 112)
    ax.set_title("(b) one setting, same architecture")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4.pdf")); fig.savefig(os.path.join(OUT, "fig4.png"))
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Fig. 5: long-tail buckets
#
# Primary source: analysis/bucket_metrics.py --json-out, which computes micro-F1 and
# macro-AUC per training-frequency bucket from the dumped predictions. The literals below
# are the values the *published* figure was drawn from, kept only so that this figure can
# still be regenerated on a machine without the long-tail dump; when they are used the
# script prints a warning, because a fallback silently used after a re-run would put stale
# numbers into a fresh figure.
# ─────────────────────────────────────────────────────────────────────────────
BUCKETS = ["N=1", "2-5", "6-20", "21-100", ">100"]

F1_FALLBACK = {"1_msmn_fp32_naive": [2.0, 6.2, 17.8, 34.0, 62.9],
               "2_msmn_bf16_naive": [0.0, 0.0, 0.2, 3.0, 51.2],
               "3_transf_bf16_naive": [0.0, 1.1, 10.0, 26.3, 57.7],
               "5_transf_bf16_varlen_fa": [0.0, 0.6, 8.7, 25.8, 57.4],
               "6_transf_fp32_naive": [0.0, 0.3, 7.2, 24.2, 57.4],
               "8_hybrid_labelfp32": [0.0, 2.7, 13.0, 30.7, 58.7]}
AUC_FALLBACK = {"1_msmn_fp32_naive": [94.0, 94.3, 95.1, 96.2, 95.6],
                "2_msmn_bf16_naive": [85.6, 85.2, 85.8, 89.0, 91.1],
                "3_transf_bf16_naive": [93.6, 94.3, 94.7, 95.5, 95.0],
                "5_transf_bf16_varlen_fa": [93.0, 93.8, 94.6, 95.5, 95.0],
                "6_transf_fp32_naive": [94.4, 93.2, 94.4, 95.4, 94.8],
                "8_hybrid_labelfp32": [92.4, 93.6, 94.3, 95.7, 95.1]}


def load_bucket_tables():
    """→ (f1_by_cell, auc_by_cell, bucket_labels, source description).

    Reads the JSON from bucket_metrics.py. Its shape is
        {"scheme": "5buckets", "buckets": ["N=1", ...],
         "models": {"<cell tag>": {"f1mi": [...], "aucm": [...], ...}}}
    and any cell it does not contain is simply left out of the figure -- better a shorter
    bar group than a bar that no run produced.
    """
    if os.path.exists(BUCKET_JSON):
        with open(BUCKET_JSON, encoding="utf-8") as f:
            d = json.load(f)
        labels = d.get("buckets") or BUCKETS
        f1, auc = {}, {}
        for tag, m in (d.get("models") or {}).items():
            if tag not in COL:                    # unknown name: no colour, no slot
                print("   [warn] bucket json has %r, which is not a known cell; skipped" % tag)
                continue
            if m.get("f1mi") is not None:
                f1[tag] = [np.nan if v is None else v for v in m["f1mi"]]
            if m.get("aucm") is not None:
                auc[tag] = [np.nan if v is None else v for v in m["aucm"]]
        if f1 or auc:
            return f1, auc, labels, "bucket_metrics json (%s)" % BUCKET_JSON

    print("   [warn] %s not found (or empty)." % BUCKET_JSON)
    print("          Fig. 5 falls back to the values the published figure was drawn from.")
    print("          Regenerate it with:  python analysis/bucket_metrics.py "
          "--json-out %s ..." % BUCKET_JSON)
    return dict(F1_FALLBACK), dict(AUC_FALLBACK), BUCKETS, "fallback constants"


def fig5():
    f1, auc, labels, src = load_bucket_tables()
    print("   fig5 data: %s (%d cells)" % (src, len(f1)))
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.9), sharex=True)
    for ax, D, ttl, ylab in [(axes[0], f1, "micro-F1 by training frequency", "micro-F1 (%)"),
                             (axes[1], auc, "AUC-macro by training frequency", "AUC-macro (%)")]:
        n = max(1, len(D)); w = 0.135 if n <= 6 else 0.8 / n
        for i, (t, v) in enumerate(D.items()):
            ax.bar(np.arange(len(labels)) + (i - (n - 1) / 2) * w, v, w,
                   color=COL[t], label=NICE[t].replace("\n", " "))
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels([l.replace("$", "").replace("\\le", "≤").replace("N >", ">")
                            for l in labels])
        ax.set_xlabel("codes per bucket (training frequency)")
        ax.set_ylabel(ylab); ax.set_title(ttl)
    # Limits follow the data instead of clipping it: the published run tops out at 62.9
    # (micro-F1) and 95.7 (AUC-macro), but a re-run must not lose a bar above that.
    def _top(D, floor):
        vals = [v for arr in D.values() for v in arr if v is not None and not np.isnan(v)]
        return max(floor, (max(vals) * 1.06) if vals else floor)

    axes[0].set_ylim(0, _top(f1, 72))
    axes[1].set_ylim(80, min(100.5, _top(auc, 100)))
    axes[0].legend(frameon=False, fontsize=4.9, ncol=2, loc="upper left",
                   handlelength=1.1, columnspacing=0.9, labelspacing=0.35)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig5.pdf")); fig.savefig(os.path.join(OUT, "fig5.png"))
    plt.close(fig)


def main():
    global MIRROR, OUT, BUCKET_JSON
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None,
                    help="results root holding runs_v2/ runs_seeds/ ... "
                         "(default: $ICD_RESULTS_ROOT, then the paper's mirror, then the repo)")
    ap.add_argument("--out", default=None, help="figure output directory (default: figures/out)")
    ap.add_argument("--bucket-json", default=None,
                    help="bucket_metrics.py --json-out file for Fig. 5 "
                         "(default: <root>/longtail/bucket_metrics.json)")
    ap.add_argument("--only", default=None,
                    help="comma-separated subset of figures to draw, e.g. 3,5 (default: all)")
    a = ap.parse_args()

    if a.root:
        MIRROR = resolve_results_root(a.root)
    if a.out:
        OUT = a.out
    if a.bucket_json:
        BUCKET_JSON = a.bucket_json
    elif not os.environ.get("ICD_BUCKET_JSON"):
        # follows the resolved root, or --root would silently keep pointing at the default
        BUCKET_JSON = os.path.join(MIRROR, "longtail", "bucket_metrics.json")
    os.makedirs(OUT, exist_ok=True)

    print("results root: %s" % MIRROR)
    print("output      : %s" % OUT)
    have_runs = os.path.isdir(os.path.join(MIRROR, "runs_v2"))
    if not os.path.isdir(MIRROR):
        print("   [warn] that directory does not exist; every data-driven figure will be empty")
    elif not have_runs:
        print("   [warn] %s has no runs_v2/; pointing --root at the wrong level is the "
              "usual cause" % MIRROR)

    want = {s.strip() for s in (a.only or "1,2,3,4,5").split(",") if s.strip()}
    figs = [("1", fig1), ("2", fig2), ("3", fig3), ("4", fig4), ("5", fig5)]
    failed = []
    for name, fn in figs:
        if name not in want:
            continue
        try:
            fn()
        except Exception as exc:                  # one missing panel must not lose the rest
            print("   [warn] fig%s could not be drawn: %s: %s"
                  % (name, type(exc).__name__, exc))
            failed.append(name)

    print("figures ->", OUT)
    for f in sorted(os.listdir(OUT)):
        print("   ", f, os.path.getsize(os.path.join(OUT, f)) // 1024, "KB")
    if failed or not have_runs:
        extra = "" if failed else "\n[warn] no run directories were found, so every " \
                                 "data-driven figure is empty."
        if failed:
            extra = ("\n[warn] incomplete: fig%s. That is expected without the corresponding"
                     " run directories; nothing was silently substituted." % ", fig".join(failed))
        print(extra)
        return 1
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

    print("figures ->", OUT)
    for f in sorted(os.listdir(OUT)):
        print("   ", f, os.path.getsize(os.path.join(OUT, f)) // 1024, "KB")
