#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""make_synthetic_runs.py -- write a fake results tree so that the CPU-side half of the
pipeline (analysis/ + figures/) can be exercised without MIMIC-III, a GPU or a checkpoint.

Why this exists
---------------
Nothing in analysis/ or figures/ needs a model; it needs *run directories in the shape the
training code leaves behind*. Without one, a reader cannot tell a broken join (a script
looking for the wrong name, or reading the wrong key) from a missing experiment, and the
first thing they discover is a KeyError. This script manufactures that shape:

    <out>/runs_v2/<hash>-_<cell>/          metric_log, epochs.jsonl, args.json
    <out>/runs_seeds/<hash>-_<cell>_s4/     full-scale extra seeds for cells 1 and 2
    <out>/runs_seeds/mimic3-50_...-<_cell>_s1..3   the 50-code seeds
    <out>/runs_diag/...-_7b_hybrid_50code_diag/
    <out>/longtail/<cell>/{y_true,y_prob,y_pred,train_freq}.npy

The numbers are synthetic and *not* the paper's: cells 1/2 carry a deliberate
fp32-vs-bf16 gap and cell 7 is made to collapse, so a figure drawn from this tree has the
right structure and obviously wrong values. Never quote them.

Output format is byte-compatible with the two producers it imitates:
  * metric_log  -- evaluation.print_metrics (the "------" / header / comma-line block)
  * epochs.jsonl -- the structured record written in model/main/main.py

Usage:
    python3 tests/make_synthetic_runs.py --out tests/out/synthetic
    python3 analysis/summarize_results.py --runs tests/out/synthetic/runs_v2 --out tests/out/summary
    python3 figures/make_figures.py --root tests/out/synthetic --out tests/out/figures
or just:  bash tests/cpu_smoke.sh
"""
import argparse
import json
import os
import random

import numpy as np

METRIC_KEYS = ["auc_macro", "auc_micro", "f1_macro", "f1_micro",
               "prec_at_5", "prec_at_8", "prec_at_15",
               "rec_at_5", "rec_at_8", "rec_at_15"]
METRIC_HEADER = "MACRO-auc, MICRO-auc, MACRO-f1, MICRO-f1, P@5, P@8, P@15, R@5, R@8, R@15"

#: One profile per cell. auc/f1 are the *final* epoch's test values; the trajectory rises
#: towards them. peak_mem and epoch_seconds are in the range the paper reports.
CELLS = {
    "1_msmn_fp32_naive":          dict(auc=0.946, mi_f1=0.610, ma_f1=0.055, p8=0.735, mem=18.5, secs=1400),
    "2_msmn_bf16_naive":          dict(auc=0.856, mi_f1=0.505, ma_f1=0.011, p8=0.628, mem=13.2, secs=1290),
    "3_transf_bf16_naive":        dict(auc=0.950, mi_f1=0.560, ma_f1=0.021, p8=0.668, mem=9.8,  secs=1330),
    "4_transf_bf16_memeff":       dict(auc=0.951, mi_f1=0.571, ma_f1=0.023, p8=0.671, mem=8.9,  secs=1300),
    "5_transf_bf16_varlen_fa":    dict(auc=0.950, mi_f1=0.573, ma_f1=0.024, p8=0.674, mem=7.4,  secs=900),
    "6_transf_fp32_naive":        dict(auc=0.952, mi_f1=0.615, ma_f1=0.052, p8=0.729, mem=15.6, secs=2560),
    # cell 7 is the documented collapse; cell 8 repairs it
    "7_hybrid_lstm_label_fa_text": dict(auc=0.620, mi_f1=0.302, ma_f1=0.004, p8=0.412, mem=6.1, secs=1100),
    "8_hybrid_labelfp32":         dict(auc=0.951, mi_f1=0.598, ma_f1=0.041, p8=0.712, mem=10.4, secs=1500),
}
SMALL = {                       # MIMIC-III-50, 200 slots: the gap almost vanishes
    "1_msmn_fp32_naive": dict(auc=0.912, mi_f1=0.702, ma_f1=0.380, p8=0.760, mem=3.1, secs=540),
    "2_msmn_bf16_naive": dict(auc=0.911, mi_f1=0.700, ma_f1=0.379, p8=0.758, mem=2.9, secs=530),
    "3_transf_bf16_naive": dict(auc=0.845, mi_f1=0.640, ma_f1=0.322, p8=0.701, mem=2.2, secs=590),
    "5_transf_bf16_varlen_fa": dict(auc=0.848, mi_f1=0.645, ma_f1=0.327, p8=0.705, mem=1.9, secs=450),
}
CELL_CFG = {
    "1_msmn_fp32_naive": dict(combiner="lstm", bf16=False, rnn_dim=256, num_layers=2,
                              attn_impl="naive", dec_chunk_codes=2048, label_repr=None),
    "2_msmn_bf16_naive": dict(combiner="lstm", bf16=True, rnn_dim=256, num_layers=2,
                              attn_impl="naive", dec_chunk_codes=2048, label_repr=None),
    "3_transf_bf16_naive": dict(combiner="transformer", bf16=True, rnn_dim=512,
                                num_transformer_layers=4, attn_impl="naive",
                                label_repr="bucketed", dec_chunk_codes=2048),
    "4_transf_bf16_memeff": dict(combiner="transformer", bf16=True, rnn_dim=512,
                                 num_transformer_layers=4, attn_impl="mem_efficient",
                                 label_repr="bucketed"),
    "5_transf_bf16_varlen_fa": dict(combiner="transformer", bf16=True, rnn_dim=512,
                                    num_transformer_layers=4, attn_impl="varlen",
                                    label_repr="varlen"),
    "6_transf_fp32_naive": dict(combiner="transformer", bf16=False, rnn_dim=512,
                                num_transformer_layers=4, attn_impl="naive",
                                label_repr="bucketed", dec_chunk_codes=2048),
    "7_hybrid_lstm_label_fa_text": dict(combiner="transformer", bf16=True, rnn_dim=512,
                                        num_transformer_layers=4, attn_impl="varlen",
                                        label_combiner="lstm", label_rnn_dim=256),
    "8_hybrid_labelfp32": dict(combiner="transformer", bf16=True, rnn_dim=512,
                               num_transformer_layers=4, attn_impl="naive",
                               label_combiner="lstm", label_rnn_dim=256, label_fp32=True,
                               label_repr="bucketed", dec_chunk_codes=2048),
}


def metric_block(profile, frac, rng):
    """One epoch's metrics as a dict, interpolated towards the profile's final values."""
    jitter = 1.0 + rng.uniform(-0.004, 0.004)
    auc = profile["auc"] * (0.86 + 0.14 * frac) * jitter
    mi = profile["mi_f1"] * (0.55 + 0.45 * frac) * jitter
    ma = profile["ma_f1"] * (0.35 + 0.65 * frac) * jitter
    p8 = profile["p8"] * (0.70 + 0.30 * frac) * jitter
    d = {"auc_macro": auc, "auc_micro": min(0.999, auc + 0.03),
         "f1_macro": ma, "f1_micro": mi,
         "prec_at_5": min(0.999, p8 + 0.08), "prec_at_8": min(0.999, p8),
         "prec_at_15": min(0.999, p8 * 0.78),
         "rec_at_5": min(0.999, p8 * 0.45), "rec_at_8": min(0.999, p8 * 0.55),
         "rec_at_15": min(0.999, p8 * 0.70)}
    return {k: round(float(v), 6) for k, v in d.items()}


def write_run_dir(path, tag, profile, epochs, seed, cfg, with_best=True):
    os.makedirs(path, exist_ok=True)
    rng = random.Random(seed)
    dev_rows, test_rows, jsonl = [], [], []
    for e in range(epochs):
        frac = (e + 1) / epochs
        dev = metric_block(profile, frac, rng)
        test = metric_block({**profile, "auc": profile["auc"] * rng.uniform(0.995, 1.005)}, frac, rng)
        dev_rows.append((e, dev))
        test_rows.append((e, test))
        jsonl.append({
            "epoch": e,
            "train_loss": round(0.0055 - 0.0015 * frac + rng.uniform(-1e-4, 1e-4), 6),
            "train_kl": 0.0,
            "train_c_loss": 0.0,
            "steps": 2975,
            "step_ms_mean": round(profile["secs"] * 1000.0 / 2975.0, 3),
            "peak_mem_GB": round(profile["mem"] * rng.uniform(0.985, 1.0), 3),
            "lr": round(5e-4 * (1 - frac) + 1e-6, 8),
            "epoch_seconds": round(profile["secs"] * rng.uniform(0.97, 1.03), 1),
            "dev": dev,
            "test": test,
        })

    with open(os.path.join(path, "metric_log"), "w", encoding="utf-8") as f:
        for e, m in dev_rows:
            f.write("------\nDev_Epoch%d\n%s\n%s\n"
                    % (e, METRIC_HEADER, ", ".join("%.4f" % m[k] for k in METRIC_KEYS)))
        for e, m in test_rows:
            f.write("------\nTest_Epoch%d\n%s\n%s\n"
                    % (e, METRIC_HEADER, ", ".join("%.4f" % m[k] for k in METRIC_KEYS)))
        if with_best:
            k = max(range(len(dev_rows)), key=lambda i: dev_rows[i][1]["prec_at_8"])
            for split, m in (("Train", {"prec_at_8": dev_rows[k][1]["prec_at_8"]}),
                             ("Dev", dev_rows[k][1]), ("Test", test_rows[k][1])):
                f.write("------\nBest_%s_Epoch%d\n%s\n%s\n"
                        % (split, k, METRIC_HEADER,
                           ", ".join("%.4f" % m.get(key, 0.0) for key in METRIC_KEYS)))

    with open(os.path.join(path, "epochs.jsonl"), "w", encoding="utf-8") as f:
        for row in jsonl:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    args = {"version": "mimic3", "decoder": "MultiLabelMultiHeadLAATV2", "attention_dim": 512,
            "truncate_length": 4000, "batch_size": 8, "learning_rate": 5e-4,
            "gradient_accumulation_steps": 1, "xavier": True, "rdrop_alpha": 5.0,
            "term_count": 4, "train_epoch": epochs, "tag": tag}
    args.update(cfg)
    with open(os.path.join(path, "args.json"), "w", encoding="utf-8") as f:
        json.dump(args, f, indent=2)
    return path


def bucket_arrays(freq_plan, n_docs, seed):
    """y_true / y_prob / y_pred whose quality tracks the training frequency of a code.

    Rare codes get a weak signal plus the same noise, so they are missed -- which is the
    shape the long-tail table is supposed to show. A fixture where every bucket scores
    100% would pass through the whole pipeline while exercising none of it.
    """
    rng = np.random.RandomState(seed)
    n_labels = len(freq_plan)
    freq = np.asarray(freq_plan, dtype=int)
    # every document carries 3-6 codes, drawn towards frequent labels
    w = freq.astype(float) / freq.sum()
    y_true = np.zeros((n_docs, n_labels), dtype=np.int8)
    for i in range(n_docs):
        k = rng.randint(3, 7)
        for j in rng.choice(n_labels, size=min(k, n_labels), replace=False, p=w):
            y_true[i, j] = 1
    # signal strength rises with frequency: that is the long tail the paper measures
    lf = np.log1p(freq).astype(float)
    q = (lf - lf.min()) / max(1e-9, (lf.max() - lf.min()))
    strength = 0.18 + 0.80 * q
    y_prob = np.clip(y_true * strength[None, :] * rng.uniform(0.55, 1.0, (n_docs, n_labels))
                     + rng.uniform(0.0, 0.62, (n_docs, n_labels)), 0, 1)
    y_pred = (y_prob > 0.45).astype(np.int8)
    return y_true, y_prob, y_pred, freq


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "out", "synthetic"))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--longtail", action="store_true", default=True,
                    help="also write the long-tail arrays (default on)")
    a = ap.parse_args()

    root = os.path.abspath(a.out)
    runs = os.path.join(root, "runs_v2")
    seeds = os.path.join(root, "runs_seeds")
    diag = os.path.join(root, "runs_diag")
    print("writing a synthetic results tree under %s" % root)

    # ---- the eight cells: name them the way run_matrix_v2.sh does (-<tag> suffix) ----
    for i, (tag, prof) in enumerate(CELLS.items(), 1):
        # a hash-like prefix, so the suffix matching that analysis relies on is exercised
        prefix = "word_0.2_lstm_2_256_0.1_MultiLabelMultiHeadLAATV2_512_xav_0.2_4_bsz16_AdamW"
        p = write_run_dir(os.path.join(runs, "%s-%s" % (prefix, tag)),
                          tag, prof, a.epochs, a.seed + i, CELL_CFG[tag])
        print("  runs_v2      %-32s %s" % (tag, os.path.basename(p)))

    # ---- extra full-scale seeds for cells 1/2 (Table 9 error bars) ----
    for j, tag in enumerate(["1_msmn_fp32_naive", "2_msmn_bf16_naive"]):
        for s in ("s4", "s5"):
            prof = dict(CELLS[tag], auc=CELLS[tag]["auc"] + 0.0007 * (1 if s == "s4" else -1))
            write_run_dir(os.path.join(seeds, "mimic3_%s_%s" % ("x", "%s_%s" % (tag, s))),
                          "%s_%s" % (tag, s), prof, a.epochs, a.seed + 100 + j, CELL_CFG[tag])
    print("  runs_seeds   full-scale s4/s5 for cells 1,2")

    # ---- the 50-code setting, three seeds (Table 9) ----
    for tag, prof in SMALL.items():
        for s in (1, 2, 3):
            jitter = 1.0 + 0.0015 * (s - 2)
            write_run_dir(os.path.join(seeds, "mimic3-50_%s" % ("%s_s%d" % (tag, s))),
                          "%s_s%d" % (tag, s),
                          {**prof, "auc": prof["auc"] * jitter, "mi_f1": prof["mi_f1"] * jitter},
                          a.epochs, a.seed + 200 + s, CELL_CFG[tag])
    print("  runs_seeds   50-code s1..s3 for cells 1,2,3,5")

    # ---- the 50-code diagnostic of cell 7 (Fig. 4b, left bar) ----
    write_run_dir(os.path.join(diag, "mimic3-50_7b_hybrid_50code_diag"),
                  "7b_hybrid_50code_diag",
                  dict(SMALL["2_msmn_bf16_naive"], auc=0.905, mi_f1=0.694),
                  3, a.seed + 300, CELL_CFG["7_hybrid_lstm_label_fa_text"])
    print("  runs_diag    7b_hybrid_50code_diag")

    # ---- long-tail arrays: five bucket sizes, so bucket_metrics fills every bucket ----
    if a.longtail:
        plan = ([1] * 10) + ([3] * 10) + ([12] * 10) + ([50] * 10) + ([400] * 10)
        for i, tag in enumerate(CELLS, 1):
            d = os.path.join(root, "longtail", tag)
            os.makedirs(d, exist_ok=True)
            y_true, y_prob, y_pred, freq = bucket_arrays(plan, 400, a.seed + 400 + i)
            np.save(os.path.join(d, "y_true.npy"), y_true)
            np.save(os.path.join(d, "y_prob.npy"), y_prob)
            np.save(os.path.join(d, "y_pred.npy"), y_pred)
            if i == 1:
                np.save(os.path.join(d, "train_freq.npy"), freq)
        print("  longtail     y_true/y_prob/y_pred for 8 cells + train_freq")

    print("\nnext:")
    print("  python3 analysis/summarize_results.py --runs %s --out %s/summary"
          % (runs, root))
    print("  python3 analysis/aggregate_seeds.py --root %s" % root)
    print("  python3 figures/make_figures.py --root %s --out %s/figures" % (root, root))
    print("\nThese numbers are synthetic. Do not quote them.")


if __name__ == "__main__":
    main()
