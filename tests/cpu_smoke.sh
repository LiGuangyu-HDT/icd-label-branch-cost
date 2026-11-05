#!/bin/bash
# =============================================================================
# cpu_smoke.sh -- check the CPU half of the pipeline end to end, with no MIMIC-III, no GPU
# and no checkpoint.
#
# What it proves, in order:
#   0. every cell's flags are defined by the model tree that cell runs from
#      (tests/check_cell_args.py) -- a typo there only fails at launch;
#   1. analysis/summarize_results.py finds run directories by the name suffix that
#      study/run_matrix_v2.sh's --tag produces, and parses metric_log / epochs.jsonl;
#   2. analysis/aggregate_seeds.py resolves a results root and reports mean +- s.d.;
#   3. analysis/bucket_metrics.py computes the five frequency buckets and writes the JSON
#      that Fig. 5 reads;
#   4. figures/make_figures.py draws Fig. 1-5 from that tree and exits 0.
#
# It cannot check the model: tests/ needs CUDA and a checkpoint, and the training steps need
# MIMIC-III. This is the part that has no excuse to be broken.
#
# Usage:
#   bash tests/cpu_smoke.sh                 # fresh tree under tests/out/synthetic
#   bash tests/cpu_smoke.sh /tmp/elsewhere  # write the scratch tree somewhere else
# =============================================================================
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
OUT="${1:-$HERE/out/synthetic}"
FAIL=0

step () { printf "\n\033[1m== %s ==\033[0m\n" "$*"; }
check () {                              # check <rc> <description>
  if [ "$1" -eq 0 ]; then echo "   ✅ $2"; else echo "   ❌ $2 (rc=$1)"; FAIL=$((FAIL+1)); fi
}

command -v python3 >/dev/null || { echo "python3 not found"; exit 2; }
python3 -c "import numpy" 2>/dev/null || { echo "numpy is required"; exit 2; }
python3 -c "import matplotlib" 2>/dev/null || { echo "matplotlib is required for figures"; exit 2; }

cd "$REPO" || exit 2

step "0/5  cell arguments vs the parsers"
python3 tests/check_cell_args.py > "$OUT.args.log" 2>&1
check $? "all 8 cells parse against model/main / model/hybrid"
grep -E "cell [78]" "$OUT.args.log" | sed 's/^/   /'

step "1/5  synthetic results tree"
python3 tests/make_synthetic_runs.py --out "$OUT" > "$OUT.mk.log" 2>&1
check $? "tests/make_synthetic_runs.py --out $OUT"
tail -3 "$OUT.mk.log" | sed 's/^/   /'

step "2/5  analysis/summarize_results.py"
python3 analysis/summarize_results.py --runs "$OUT/runs_v2" --out "$OUT/summary" \
    > "$OUT.summarize.log" 2>&1
rc=$?; check $rc "summarize_results: 8 cells, tables.md + tables.tex + epochs.csv"
if [ $rc -eq 0 ]; then
  # The join is the whole point: a "not started" cell here means a name mismatch.
  n_done=$(grep -c "done " "$OUT/summary/SUMMARY.txt" 2>/dev/null || echo 0)
  if [ "$n_done" -ge 8 ]; then echo "   ✅ all 8 cells found (suffix join works)"
  else echo "   ❌ only $n_done/8 cells found -- the run-directory name does not match"; FAIL=$((FAIL+1)); fi
fi

step "3/5  analysis/aggregate_seeds.py"
python3 analysis/aggregate_seeds.py --root "$OUT" > "$OUT.aggregate.log" 2>&1
check $? "aggregate_seeds --root $OUT"
grep -E "n=3 +AUC-macro" "$OUT.aggregate.log" | head -2 | sed 's/^/   /'

step "4/5  analysis/bucket_metrics.py -> JSON, then figures/make_figures.py"
MODELS=()
for d in "$OUT"/longtail/*/; do
  tag=$(basename "$d")
  [ -f "$d/y_true.npy" ] || continue
  MODELS+=(--model "$tag" "$d/y_true.npy" "$d/y_prob.npy" "$d/y_pred.npy")
done
python3 analysis/bucket_metrics.py "${MODELS[@]}" \
    --train-freq "$OUT/longtail/1_msmn_fp32_naive/train_freq.npy" \
    --out "$OUT/latex_rows.tex" --json-out "$OUT/longtail/bucket_metrics.json" \
    > "$OUT.buckets.log" 2>&1
check $? "bucket_metrics: 5 buckets x $(( ${#MODELS[@]} / 5 )) cells, JSON written"

python3 figures/make_figures.py --root "$OUT" --out "$OUT/figures" > "$OUT.figures.log" 2>&1
check $? "make_figures: fig1..fig5 from that tree"
grep -E "fig5 data|results root" "$OUT.figures.log" | sed 's/^/   /'

echo
if [ "$FAIL" -eq 0 ]; then
  echo "ALL CPU CHECKS PASSED  ->  $OUT"
  echo "figures: $OUT/figures   tables: $OUT/summary/tables.md"
  exit 0
else
  echo "$FAIL CHECK(S) FAILED"
  echo "logs: $OUT.*.log"
  exit 1
fi
