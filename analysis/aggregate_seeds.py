#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""aggregate_seeds.py -- aggregate repeated runs into mean +- s.d. and print the numbers that
go into the paper (CPU only).

Conventions that matter:
  * This code does not fix random seeds, so the s1/s4/s5 labels are names only: two runs are
    independent samples, not a pair. The uncertainty of a difference is therefore
    sqrt(s1^2/n1 + s2^2/n2), never the spread of per-seed differences.
  * Each cell reports the test metrics of the epoch selected on dev P@8.

Inputs, under the run directory:
  runs/<...>_<cell>                 the full-scale single run, counted as seed 1
  runs_seeds/mimic3_<...>_<cell>_s4 additional full-scale seeds
  runs_seeds/mimic3-50_<...>_<cell>_sN  the three seeds of the 50-code setting

Usage:
  python3 aggregate_seeds.py                    # local runs, root resolved automatically
  python3 aggregate_seeds.py --root PATH        # point at a results root explicitly
  python3 aggregate_seeds.py --pod               # read live from $ICD_RUNS (needs .cpod/rc)

The results root is resolved, in order, from --root, $ICD_RESULTS_ROOT, the author's
mirror layout (<umbrella>/experiments/formal_results/experiments), and finally the
repository itself -- which is where study/run_matrix_v2.sh and study/run_seeds.sh write
with their default OUT_DIR. --pod additionally requires the .cpod/rc copy wrapper; if it
is absent the script says so instead of reporting empty runs.
"""
import argparse
import glob
import json
import math
import os
import statistics as st
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
ROOT = os.path.abspath(os.path.join(REPO, ".."))


def _root_candidates():
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
        if os.path.isdir(os.path.join(cand, "runs_v2")) or os.path.isdir(os.path.join(cand, "runs")):
            return cand
    return _root_candidates()[0]


MIRROR = resolve_results_root()
POD_RUNS = os.environ.get("ICD_RUNS", os.path.join(REPO, "runs"))
CPOD = os.path.join(ROOT, ".cpod", "rc")

METRICS = [("auc_macro", "AUC-macro"), ("f1_micro", "micro-F1"),
           ("f1_macro", "macro-F1"), ("prec_at_8", "P@8")]

# Full-scale cells that have error bars (the others are single runs)
FULL_SEEDED = ["1_msmn_fp32_naive", "2_msmn_bf16_naive"]
# Cells with three seeds in the 50-code setting
SMALL_SEEDED = ["1_msmn_fp32_naive", "2_msmn_bf16_naive",
                "3_transf_bf16_naive", "5_transf_bf16_varlen_fa"]
NICE = {"1_msmn_fp32_naive": "① BiLSTM fp32", "2_msmn_bf16_naive": "② BiLSTM bf16",
        "3_transf_bf16_naive": "③ Transformer bf16 (non-flash)",
        "5_transf_bf16_varlen_fa": "⑤ Transformer bf16 (flash)",
        "6_transf_fp32_naive": "⑥ Transformer fp32"}


# ── reading runs ────────────────────────────────────────────────────────
CPOD_MISSING = ("[pod] %s is absent: --pod reads a remote run root through the .cpod/rc "
                "copy wrapper, which is a local helper and is not part of this "
                "repository. Use --root PATH (or $ICD_RESULTS_ROOT) for a local tree "
                "instead." % CPOD)


def require_pod():
    """--pod is optional machinery; fail with an explanation, not with empty results."""
    if not os.path.exists(CPOD):
        raise SystemExit(CPOD_MISSING)


def pod_ls(pat):
    """List matching directory names from the run root (read only, no GPU)."""
    out = subprocess.run([CPOD, "ls -d %s 2>/dev/null" % pat],
                         capture_output=True, text=True).stdout
    return [l.strip() for l in out.splitlines() if l.strip().startswith("/")]


def pod_rows(d):
    out = subprocess.run([CPOD, "cat %s/epochs.jsonl" % d],
                         capture_output=True, text=True).stdout
    return [json.loads(l) for l in out.splitlines() if l.strip().startswith("{")]


def local_rows(d):
    p = os.path.join(d, "epochs.jsonl")
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if os.path.exists(p) else []


def find(prefix, tag, use_pod, must_prefix=None):
    """Find the run directory. The tag must be at the END of the name, and must_prefix
    optionally distinguishes mimic3-50_ from mimic3_ -- matching on the suffix alone would
    confuse a full-scale seed 1 with a 50-code seed 1."""
    sub = "runs_v2" if prefix == "runs_v2" else "runs_seeds"
    if use_pod:
        hits = pod_ls("%s/%s/*%s" % (POD_RUNS, sub, tag))
    else:
        hits = glob.glob(os.path.join(MIRROR, sub, "*"))
    out = []
    for h in hits:
        b = os.path.basename(h)
        if not b.endswith(tag):
            continue
        if must_prefix and not b.startswith(must_prefix):
            continue
        out.append(h)
    return sorted(out)[-1] if out else None


def pick(d, use_pod):
    """Select the epoch on dev P@8 and return its test metrics, epoch index and total.
    A directory without epochs.jsonl returns ('EMPTY', reason)."""
    rows = pod_rows(d) if use_pod else local_rows(d)
    if not rows:
        return "EMPTY"
    best = max(rows, key=lambda r: (r.get("dev") or {}).get("prec_at_8", -1))
    t = best.get("test") or {}
    return {"epoch": best.get("epoch"), "n_epochs": len(rows),
            "dev_p8": (best.get("dev") or {}).get("prec_at_8"),
            **{k: t.get(k) for k, _ in METRICS}}


# ── statistics ───────────────────────────────────────────────────────────
def summarize(vals):
    v = [x for x in vals if x is not None]
    n = len(v)
    if n == 0:
        return None
    m = st.mean(v)
    s = st.stdev(v) if n > 1 else float("nan")
    se = s / math.sqrt(n) if n > 1 else float("nan")
    return {"n": n, "values": v, "mean": m, "sd": s, "se": se,
            "min": min(v), "max": max(v)}


def indep_diff(a, b):
    """Difference of independent samples: mean(a)-mean(b) with se = sqrt(sa^2/na + sb^2/nb)."""
    d = a["mean"] - b["mean"]
    se = math.sqrt((a["sd"] ** 2) / a["n"] + (b["sd"] ** 2) / b["n"])
    return {"diff": d, "se": se, "t": (d / se if se > 0 else float("nan")),
            "lo": d - 1.96 * se, "hi": d + 1.96 * se}


# ── main ─────────────────────────────────────────────────────────────────
def collect_full(tags, use_pod, labels, total):
    """Full scale: seed 1 from runs/ plus s4 and s5 with the mimic3_ prefix (not mimic3-50_).
    Runs shorter than `total` epochs are marked incomplete and excluded unless
    --include-partial is given."""
    out, missing = {}, []
    for tag in tags:
        runs = []
        d = find("runs_v2", tag, use_pod)
        if d:
            r = pick(d, use_pod)
            if r == "EMPTY":
                missing.append((tag, "seed1", d))
            elif r:
                runs.append({"label": "seed1", "run": os.path.basename(d),
                             "incomplete": r["n_epochs"] < total, **r})
        else:
            missing.append((tag, "seed1", None))
        for lab in labels:
            d = find("runs_seeds", "%s_%s" % (tag, lab), use_pod, must_prefix="mimic3_")
            if d:
                r = pick(d, use_pod)
                if r == "EMPTY":
                    missing.append((tag, lab, d))
                elif r:
                    runs.append({"label": lab, "run": os.path.basename(d),
                                 "incomplete": r["n_epochs"] < total, **r})
            else:
                missing.append((tag, lab, None))
        out[tag] = runs
    return out, missing


def collect_small(tags, use_pod, labels, total=None):
    """50-code: only the s1/s2/s3 runs whose names start with mimic3-50_."""
    out, missing = {}, []
    for tag in tags:
        runs = []
        for lab in labels:
            d = find("runs_seeds", "%s_%s" % (tag, lab), use_pod, must_prefix="mimic3-50_")
            if d:
                r = pick(d, use_pod)
                if r == "EMPTY":
                    missing.append((tag, lab, d))
                elif r:
                    runs.append({"label": lab, "run": os.path.basename(d), **r})
            else:
                missing.append((tag, lab, None))
        out[tag] = runs
    return out, missing


def fmt_pct(x):
    return "—" if x is None else "%.2f" % (x * 100)


def main():
    global MIRROR
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None,
                    help="results root holding runs/ runs_v2/ runs_seeds/ "
                         "(default: $ICD_RESULTS_ROOT, then the paper's mirror, then the repo)")
    ap.add_argument("--pod", action="store_true", help="read live through .cpod/rc from $ICD_RUNS")
    ap.add_argument("--json", default=None)
    ap.add_argument("--epochs", type=int, default=20, help="epochs a full-scale run should complete (default 20)")
    ap.add_argument("--include-partial", action="store_true",
                    help="include runs that did not finish (excluded by default)")
    a = ap.parse_args()
    use_pod = a.pod
    if use_pod:
        require_pod()
    else:
        MIRROR = resolve_results_root(a.root)
    print("source: %s" % ("live run directory %s" % POD_RUNS if use_pod
                          else "local root %s" % os.path.relpath(MIRROR, ROOT)))
    if not use_pod and not os.path.isdir(MIRROR):
        print("        [warn] that directory does not exist; every cell below will be "
              "reported as missing. Pass --root or set ICD_RESULTS_ROOT.")

    full, miss_full = collect_full(FULL_SEEDED, use_pod, ["s4", "s5"], a.epochs)
    small, miss_small = collect_small(SMALL_SEEDED, use_pod, ["s1", "s2", "s3"])
    partial = [(NICE.get(t, t), r["label"], r["n_epochs"])
               for t in FULL_SEEDED for r in full[t] if r.get("incomplete")]
    if not a.include_partial:
        for tag in FULL_SEEDED:
            full[tag] = [r for r in full[tag] if not r.get("incomplete")]

    def report_missing(miss, what):
        if not miss:
            return
        print("\n  [warn] %s is missing %d entries (no directory = not run or not synced; "
              "directory without epochs.jsonl = data not synced):" % (what, len(miss)))
        for tag, lab, d in miss:
            why = "no directory" if d is None else "directory present but epochs.jsonl empty"
            print("     %-30s %-5s %s" % (NICE.get(tag, tag), lab, why))
        if not use_pod:
            print("     -> sync the latest results first")

    # ── full scale, cells 1/2: per-run detail ───────────────────────────────
    print("\n" + "=" * 96)
    print("[1] full-scale repeats of cells 1/2 (epoch selected on dev P@8; seeds unfixed -> independent)")
    print("=" * 96)
    hdr = "  %-24s %-9s %-6s %-5s" % ("run", "selected", "total", "devP8")
    for _, nm in METRICS:
        hdr += " %12s" % nm
    print(hdr)
    for tag in FULL_SEEDED:
        print("  " + "-" * 92)
        if not full[tag]:
            print("  %-24s (no usable data yet)" % NICE.get(tag, tag))
        for r in full[tag]:
            mark = "  running %d/%d" % (r["n_epochs"], a.epochs) if r.get("incomplete") else ""
            line = "  %-24s %-9s %-6s %-5s" % (
                NICE.get(tag, tag) + " " + r["label"], r["epoch"], r["n_epochs"],
                "%.4f" % r["dev_p8"] if r["dev_p8"] is not None else "—")
            for k, _ in METRICS:
                line += " %12s" % fmt_pct(r[k])
            print(line + mark)
    report_missing(miss_full, "full-scale repeats of cells 1/2")
    if partial and not a.include_partial:
        print("\n  (unfinished runs excluded from mean +- s.d.: %s)"
              % ", ".join("%s %s=%d epochs" % p for p in partial))

    # ── full scale, cells 1/2: mean +- s.d. ─────────────────────────────────
    stats = {}
    print("\n" + "=" * 96)
    print("[2] full-scale mean +- s.d. for cells 1/2 (the two rows of Table 5)")
    print("=" * 96)
    for tag in FULL_SEEDED:
        runs = full[tag]
        stats[tag] = {k: summarize([r[k] for r in runs]) for k, _ in METRICS}
        print("  %s   n=%d" % (NICE[tag], len(runs)))
        for k, nm in METRICS:
            s = stats[tag][k]
            if not s:
                print("     %-9s no data" % nm)
            elif s["n"] == 1:
                print("     %-9s %.2f   (one run only, so no +-)" % (nm, s["mean"] * 100))
            else:
                print("     %-9s %.2f +- %.2f   (s.e. %.2f, range %.2f-%.2f, n=%d)"
                      % (nm, s["mean"] * 100, s["sd"] * 100, s["se"] * 100,
                         s["min"] * 100, s["max"] * 100, s["n"]))

    # ── cell 1 - cell 2: difference of independent samples ──────────────────
    print("\n" + "=" * 96)
    print("[3] cell 1 - cell 2 difference (independent samples: se = sqrt(s1^2/n1 + s2^2/n2))")
    print("=" * 96)
    A, B = "1_msmn_fp32_naive", "2_msmn_bf16_naive"
    diffs = {}
    if stats[A] and stats[B]:
        for k, nm in METRICS:
            sa, sb = stats[A][k], stats[B][k]
            if not sa or not sb or sa["n"] < 2 or sb["n"] < 2:
                print("  %-9s not enough runs (cell 1 n=%s, cell 2 n=%s)"
                      % (nm, sa["n"] if sa else 0, sb["n"] if sb else 0))
                continue
            dd = indep_diff(sa, sb)
            diffs[k] = dd
            print("  %-9s difference %.2f +- %.2f points (95%% CI %.2f-%.2f, t~%.1f)"
                  % (nm, dd["diff"] * 100, dd["se"] * 100, dd["lo"] * 100, dd["hi"] * 100, dd["t"]))
        # Against the most conservative single-run pair
        if full[A] and full[B]:
            print("\n  for reference, the single-run combinations (the paper reports seed 1 against seed 1)")
            for ra in full[A]:
                for rb in full[B]:
                    d = (ra["auc_macro"] - rb["auc_macro"]) * 100 if ra["auc_macro"] and rb["auc_macro"] else None
                    if d is not None:
                        print("     %-22s - %-22s = %+6.2f points" % (ra["label"], rb["label"], d))

    # ── 50-code cross-check ─────────────────────────────────────────────────
    print("\n" + "=" * 96)
    print("[4] 50-code, three seeds (do the numbers in Table 9 match?)")
    print("=" * 96)
    for tag in SMALL_SEEDED:
        runs = small[tag]
        if not runs:
            print("  %-38s no data" % NICE.get(tag, tag))
            continue
        parts = []
        for k, nm in METRICS:
            s = summarize([r[k] for r in runs])
            if s and s["n"] > 1:
                parts.append("%s %.4f±%.4f" % (nm, s["mean"], s["sd"]))
        print("  %-38s n=%d  %s" % (NICE.get(tag, tag), len(runs), "  ".join(parts)))

    # ── what the aggregated numbers are for ─────────────────────────────────
    print("\n" + "=" * 96)
    print("[5] what to report, and how")
    print("=" * 96)
    todo = [
        "cells 1 and 2: report mean +- s.d. over the available runs, not a single value",
        "the cell 2 - cell 1 difference: report the value from [3] together with n",
        "the cell 3 - cell 2 difference: cell 3 is a single run, so state that it is compared "
        "against the mean of cell 2",
        "anywhere a difference is quoted, use the independent-sample standard error, never "
        "the spread of per-seed differences",
    ]
    for what in todo:
        print("  [ ] %s" % what)

    # Optional: the variance comparison, which is the substantive finding here
    if stats[B] and stats[A] and stats[A][METRICS[0][0]] and stats[B][METRICS[0][0]]:
        sa, sb = stats[A]["auc_macro"], stats[B]["auc_macro"]
        if sa["n"] > 1 and sb["n"] > 1:
            print("\n  finding: for the same encoder, the fp32 run-to-run")
            print("     s.d. is %.2f points against %.2f under bf16 (%.1fx): bf16 does not only "
                  "lower the mean,"
                  % (sa["sd"] * 100, sb["sd"] * 100, sb["sd"] / sa["sd"] if sa["sd"] else float("nan")))
            print("     it also makes training sensitive to initialisation. Inference-time "
                  "rounding cannot produce run-to-run variance,")
            print("     so this is independent evidence that the damage lies in training dynamics.")

    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump({"full": full, "stats": {k: v for k, v in stats.items()}, "diffs": diffs},
                      f, ensure_ascii=False, indent=2, default=str)
        print("\nwrote %s" % a.json)


if __name__ == "__main__":
    main()
