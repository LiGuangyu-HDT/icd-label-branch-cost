#!/bin/bash
# =============================================================================
# run_seeds.sh -- multi-seed runner for the error bars and the second setting
#
# Error bars are expected at this level, but a full-scale cell costs 6-72 GPU-hours, so
# three seeds there is not affordable. In the 50-code setting the training set is 8,045
# notes and there are 200 label descriptions, which brings a run down to a few hours, so
# three seeds per configuration is entirely reasonable -- and those seeds are what the
# paper's error bars come from.
#
# Where the seeds come from: nowhere. Model initialisation uses unseeded np.random and the
# DataLoader shuffles, so running the same command N times already yields N independent
# samples. Fixing a seed would artificially remove the real variance.
#
# Usage
#   bash run_seeds.sh --version mimic3-50 --cells 1,3,5 --seeds 3
#   bash run_seeds.sh --version mimic3-50 --cells 1,2,3 --seeds 1 --gpus 0
#   bash run_seeds.sh --dry        # 3 steps per cell, as a pre-check
#   bash run_seeds.sh --list       # print the plan only
#
# Requires the 50-code setting to be prepared first (prepare_mimic3_50.py).
# =============================================================================
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

VERSION="mimic3-50"
CELLS="1,3,5"
SEEDS=3
SEEDLABELS=""          # --seed-labels "4 5" runs only those seed indices
GPUS="0 1"
DRY=0
LIST=0
BS=16
EPOCHS=20
STEPS=0
MIN_FREE_GB=4
DATA_MIMIC3="${ICD_DATA_DIR:?set ICD_DATA_DIR to the prepared data directory}"
DATA_50="${ICD_DATA_50:-$HERE/../data/mimic3_50}"
WE="${ICD_WORD_EMBEDDING:?set ICD_WORD_EMBEDDING to the word2vec model}"
REPO="${ICD_REPO:-$HERE/../model/main}"
RES="${OUT_DIR:-$HERE/out/seeds}"
RUNS="${RUNS_DIR:-$HERE/out/runs_seeds}"
BK="${BACKUP_DIR:-$HERE/out/seeds_backup}"

while [ $# -gt 0 ]; do
  case "$1" in
    --version) VERSION="$2"; shift ;;
    --cells)   CELLS="$2"; shift ;;
    --seeds)   SEEDS="$2"; shift ;;
    --seed-labels) SEEDLABELS="$2"; shift ;;
    --gpus)    GPUS="$2"; shift ;;
    --steps)   STEPS="$2"; shift ;;
    --dry)     DRY=1; STEPS=3; EPOCHS=1 ;;
    --list)    LIST=1 ;;
    *) echo "unknown argument $1"; exit 1 ;;
  esac
  shift
done

if [ "$VERSION" = "mimic3-50" ]; then DATA_DIR="$DATA_50"; else DATA_DIR="$DATA_MIMIC3"; fi

cd "$REPO"
export ICD_DATA_DIR="$DATA_DIR" ICD_WORD_EMBEDDING="$WE"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8

COMMON="--version $VERSION --decoder MultiLabelMultiHeadLAATV2 \
 --attention_dim 512 --truncate_length 4000 --batch_size $BS \
 --learning_rate 5e-4 --gradient_accumulation_steps 1 --xavier \
 --main_code_loss_weight 0.0 --rdrop_alpha 5.0 --est_cls 1 \
 --word_embedding_path $WE --train_epoch $EPOCHS --n_gpu 1 \
 --output_base_dir $RUNS --save_every 1 --keep_epochs 10 --log_every 50"
BUCKETED="--label_repr bucketed --enc_chunk_tokens 16384 --enc_attn_backend efficient"

cell_args () {
  case "$1" in
    1) echo "--combiner lstm --rnn_dim 256 --num_layers 2 --attention_head 4 --term_count 4 --dec_chunk_codes 2048" ;;
    2) echo "--combiner lstm --rnn_dim 256 --num_layers 2 --attention_head 4 --term_count 4 --dec_chunk_codes 2048 --bf16" ;;
    3) echo "--combiner transformer --rnn_dim 512 --num_transformer_layers 4 --pos_embed learn --attention_head 4 --term_count 4 $BUCKETED --dec_chunk_codes 2048 --bf16" ;;
    4) echo "--combiner transformer --rnn_dim 512 --num_transformer_layers 4 --pos_embed learn --attention_head 4 --term_count 4 $BUCKETED --attn_impl mem_efficient --bf16" ;;
    5) echo "--combiner transformer --rnn_dim 512 --num_transformer_layers 4 --pos_embed learn --attention_head 4 --term_count 4 --label_repr varlen --enc_chunk_tokens 16384 --attn_impl varlen --bf16" ;;
    6) echo "--combiner transformer --rnn_dim 512 --num_transformer_layers 4 --pos_embed learn --attention_head 4 --term_count 4 $BUCKETED --dec_chunk_codes 2048" ;;
  esac
}
cell_name () {
  case "$1" in
    1) echo "1_msmn_fp32_naive" ;; 2) echo "2_msmn_bf16_naive" ;;
    3) echo "3_transf_bf16_naive" ;; 4) echo "4_transf_bf16_memeff" ;;
    5) echo "5_transf_bf16_varlen_fa" ;; 6) echo "6_transf_fp32_naive" ;;
  esac
}
# Estimated hours for 20 epochs on one device (measured per setting)
cell_est () {
  if [ "$VERSION" = "mimic3-50" ]; then
    case "$1" in 1) echo 3.0;; 2) echo 3.0;; 3) echo 3.3;; 4) echo 3.4;; 5) echo 2.5;; 6) echo 4.5;; esac
  else
    case "$1" in 1) echo 39.8;; 2) echo 35.6;; 3) echo 37.2;; 4) echo 38.0;; 5) echo 24.9;; 6) echo 71.8;; esac
  fi
}

status_write () {
  { echo "updated: $(date '+%F %T')"
    echo "setting: $VERSION | cells: $CELLS | seeds: $SEEDS | $([ "$DRY" = 1 ] && echo pre-check || echo full)"
    echo; cat "$RES/SUMMARY.txt" 2>/dev/null; } > "$RES/STATUS.txt.tmp" \
    && mv "$RES/STATUS.txt.tmp" "$RES/STATUS.txt"
}

archive_one () {
  local name="$1" d="$BK/$name" rd
  mkdir -p "$d"
  cp -f "$RES/$name.log" "$d/$name.log" 2>/dev/null || true
  cp -f "$RES/$name.log.rc" "$d/$name.log.rc" 2>/dev/null || true
  for rd in "$RUNS"/*"$name"; do
    [ -d "$rd" ] || continue
    cp -f "$rd/epochs.jsonl" "$rd/metric_log" "$rd/args.json" "$d/" 2>/dev/null || true
    break
  done
  cp -f "$RES/SUMMARY.txt" "$BK/" 2>/dev/null || true
}

run_one () {                  # run_one <cell> <seed> <GPU>
  local c="$1" s="$2" gpu="$3"
  local name; name="$(cell_name "$c")_s$s"
  local log="$RES/$name.log"
  local t0=$SECONDS

  if [ -f "$log.rc" ] && [ "$(cat "$log.rc" 2>/dev/null)" = "0" ] && [ "$STEPS" = "0" ]; then
    echo "[lane$gpu] $name already done, skipped"; return 0
  fi
  local free; free=$(df -BG "$RUNS" 2>/dev/null | tail -1 | awk '{print $4}' | tr -dc '0-9')
  if [ -n "${free:-}" ] && [ "$free" -lt "$MIN_FREE_GB" ]; then
    echo "[lane$gpu] [warn] only ${free}G free, skipping $name"
    echo "$name|SKIP_DISK" >> "$RES/SUMMARY.txt"; return 1
  fi

  local extra=""
  [ "$STEPS" -gt 0 ] && extra="--max_steps $STEPS"
  echo "[lane$gpu] ===== $name start $(date '+%F %T') est. $(cell_est "$c") h ====="
  CUDA_VISIBLE_DEVICES=$gpu python -u main.py $COMMON $(cell_args "$c") $extra \
      --tag "$name" > "$log" 2>&1
  local rc=$?
  echo "$rc" > "$log.rc"
  local dt=$((SECONDS - t0))
  local last; last=$(grep -oE '"epoch": [0-9]+' "$RUNS"/*"$name"/epochs.jsonl 2>/dev/null | tail -1 | grep -oE '[0-9]+$')
  local step; step=$(grep -oE "step [0-9.]+ ms" "$log" | tail -1)
  echo "[lane$gpu] $name done rc=$rc elapsed ${dt}s epoch=${last:-0} ${step:-}"
  echo "$name|rc=$rc|${dt}s|epochs=${last:-0}|${step:-?}" >> "$RES/SUMMARY.txt"
  status_write; archive_one "$name"
}

# ---- Build the job list and balance it across two lanes by estimated cost ----
JOBS=()
SEEDN="$SEEDLABELS"
[ -z "$SEEDN" ] && SEEDN="$(seq 1 "$SEEDS")"
for c in $(echo "$CELLS" | tr ',' ' '); do
  for s in $SEEDN; do
    JOBS+=("$c:$s:$(cell_est "$c")")
  done
done
# Sort by estimated cost, descending (LPT greedy, near-optimal makespan)
IFS=$'\n' SORTED=($(for j in "${JOBS[@]}"; do echo "$j"; done | sort -t: -k3 -nr)); unset IFS
L0=(); L1=(); T0=0; T1=0
for j in "${SORTED[@]}"; do
  t=$(echo "$j" | cut -d: -f3)
  if awk "BEGIN{exit !($T0 <= $T1)}"; then L0+=("$j"); T0=$(awk "BEGIN{print $T0+$t}"); \
  else L1+=("$j"); T1=$(awk "BEGIN{print $T1+$t}"); fi
done

echo "==============================================================="
echo "setting $VERSION | cells $CELLS | seed indices: ${SEEDN} | ${#JOBS[@]} jobs"
echo "  data directory: $DATA_DIR"
echo "  GPU0: $(for j in "${L0[@]}"; do echo -n "$(echo $j|cut -d: -f1)s$(echo $j|cut -d: -f2) "; done)  ≈ ${T0} h"
echo "  GPU1: $(for j in "${L1[@]}"; do echo -n "$(echo $j|cut -d: -f1)s$(echo $j|cut -d: -f2) "; done)  ≈ ${T1} h"
echo "  makespan ≈ $(awk "BEGIN{print ($T0>$T1)?$T0:$T1}") h"
echo "==============================================================="
if [ "$LIST" = "1" ]; then exit 0; fi
mkdir -p "$RES" "$RUNS" "$BK"
if [ ! -d "$DATA_DIR" ]; then
  echo "FAIL data directory not found: $DATA_DIR"; exit 1
fi

echo "start: $(date '+%F %T') | $VERSION | cells=$CELLS | seeds=$SEEDS | GPUs=$GPUS" >> "$RES/SUMMARY.txt"
status_write

lane () {
  local gpu="$1"; shift
  for j in "$@"; do
    run_one "$(echo $j|cut -d: -f1)" "$(echo $j|cut -d: -f2)" "$gpu"
  done
  echo "[lane$gpu] all done $(date '+%F %T')"
}
# --gpus "0" means single-device mode: every job runs on the first GPU
set -- $GPUS
G0="$1"; G1="${2:-}"
if [ -z "$G1" ]; then
  echo "  [note] single-device mode: all ${#JOBS[@]} jobs serialised on GPU${G0} (est. $(awk "BEGIN{print $T0+$T1}") h)"
  L0=("${SORTED[@]}"); L1=()
fi
lane "$G0" "${L0[@]}" & P0=$!
if [ -n "$G1" ]; then lane "$G1" "${L1[@]}" & P1=$!; else P1=""; fi
wait $P0; R0=$?
if [ -n "$P1" ]; then wait $P1; R1=$?; else R1=0; fi
echo "all done: $(date '+%F %T') | lane0 rc=$R0 lane1 rc=$R1" >> "$RES/SUMMARY.txt"
status_write
echo; echo "================ SUMMARY ================"; cat "$RES/SUMMARY.txt"
echo; echo "error-bar summary:"
echo "  python3 experiments/scripts/summarize_results.py --runs $RUNS --out $RES/summary"
