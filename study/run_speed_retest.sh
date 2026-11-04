#!/bin/bash
# =============================================================================
# run_speed_retest.sh -- speed re-measurement without contention
#
# The 20-epoch runs were measured with two cells sharing one node, so each cell was
# competing with the other for CPU. The speed numbers in the paper therefore come from
# this script instead: single-task, same-process measurements only. It trains nothing and
# writes only text logs.
#
# What it runs:
#   1) bench_encoders.py --B 16 --L 4000   text branch, BiLSTM vs Transformer (fp32/bf16)
#      (the original used B=8; the real setting here is B=16)
#   2) bench_bucketed.py --mode perf       label branch, bucketed bf16 vs fp32
#      (needs real label descriptions, via ICD_DATA_DIR)
#
# Usage:
#   export ICD_DATA_DIR=/path/to/mimic3_caml
#   export ICD_WORD_EMBEDDING=/path/to/word2vec_sg0_100.model
#   bash run_speed_retest.sh --gpu 0
# =============================================================================
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU=0
while [ $# -gt 0 ]; do
  case "$1" in --gpu) GPU="$2"; shift ;; *) echo "unknown argument $1"; exit 1 ;; esac
  shift
done

DATA_DIR="${ICD_DATA_DIR:?set ICD_DATA_DIR to the prepared data directory}"
WE="${ICD_WORD_EMBEDDING:?set ICD_WORD_EMBEDDING to the word2vec model}"
REPO="${ICD_REPO:-$HERE/../model/main}"
BENCH="$HERE/../bench"
OUT="${OUT_DIR:-$HERE/out/speed_retest}"
mkdir -p "$OUT"

cd "$REPO" || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
echo "[speed] start $(date '+%F %T') | GPU$GPU | code $REPO"

# 1) text branch A/B (synthetic data, the cleanest comparison)
echo "[speed] (1/2) bench_encoders --B 16 --L 4000"
CUDA_VISIBLE_DEVICES=$GPU python -u "$BENCH/bench_encoders.py" --B 16 --L 4000 --dtype both \
  > "$OUT/encoders_B16.log" 2>&1 || echo "[speed] [warn] bench_encoders failed (non-fatal)"

# 2) label branch, bucketed: bf16 vs fp32 (needs real descriptions)
echo "[speed] (2/2) bench_bucketed --mode perf --reprs bucketed"
CUDA_VISIBLE_DEVICES=$GPU python -u "$BENCH/bench_bucketed.py" --mode perf --dtype both \
  --reprs bucketed --chunk_tokens 16384 --sdpa_backend efficient \
  > "$OUT/bucketed_perf.log" 2>&1 || echo "[speed] [warn] bench_bucketed failed (non-fatal)"

echo "[speed] done $(date '+%F %T'); output in $OUT/"
echo "================ encoders_B16 ================"; tail -30 "$OUT/encoders_B16.log" 2>/dev/null
echo "================ bucketed_perf ==============="; tail -30 "$OUT/bucketed_perf.log" 2>/dev/null
