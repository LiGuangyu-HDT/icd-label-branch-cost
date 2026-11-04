#!/bin/bash
# =============================================================================
# run_pipeline2.sh -- the orchestrated pipeline that produced the reported runs
#
# Differences from the earlier pipeline:
#   1. The warm-up probe is dropped from the chain. The existing data already answers the
#      "not trained long enough" objection: at epoch 20 cell 3's training loss (0.004114) is
#      lower than cell 1's (0.004188) while its long-tail micro-F1 is 5.25 points worse, and
#      the AUC gap shrinks from 13.6 to 0.47 points while the micro-F1 gap stays at 5.2-5.7.
#      The script is kept for revisions, with the double-lock bug fixed.
#   2. The full-scale precision probe is added. The 50-code probe cannot resolve the
#      difference (four runs within 0.7 points) because that setting has 200 label
#      descriptions against 35,684, a factor of 180. Four runs x 3 epochs at full scale
#      localise the damage instead.
#
# The chain, on two GPUs:
#   GPU0  full-scale precision probe -> mimic3-50 single-GPU seeds -> speed re-measurement
#   GPU1  the remaining cells
#
# Each stage is waited on by file, not by process name.
# =============================================================================
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS=$HERE/../bench
RES="${OUT_DIR:-$HERE/out/pipeline2}"
PROBES=$RES/probes
RUNS="${RUNS_DIR:-$HERE/out/runs_v2}"
LOG="$RES/PIPELINE2_MAIN.log"
GPU0_FREE=/tmp/gpu0_free_v2
DONE=/tmp/pipeline2_done
PROBE_RC="$PROBES/PROBE_MAIN.log.rc"
C7_RC="$RES/formal_v2/7_hybrid_lstm_label_fa_text.log.rc"

log() { echo "[pipe2 $(date -u -d '+8 hour' '+%m-%d %H:%M')] $*" >> "$LOG"; }
wait_rc() {
  local f="$1" lim="$2" n=0
  while :; do
    if [ -f "$f" ]; then
      [ "$(cat "$f" 2>/dev/null)" = "0" ] && { log "✔ $(basename "$f") rc=0"; return 0; }
      log "[warn] $(basename "$f") rc=$(cat "$f") (non-zero, continuing)"; return 1
    fi
    sleep 60; n=$((n+1))
    [ "$n" -ge "$lim" ] && { log "[warn] timed out waiting for $(basename "$f") (${lim} min)"; return 1; }
  done
}
wait_file() {
  local f="$1" lim="$2" n=0
  while [ ! -f "$f" ]; do
    sleep 60; n=$((n+1))
    [ "$n" -ge "$lim" ] && { log "[warn] timed out waiting for $f (${lim} min)"; return 1; }
  done
  return 0
}

lane0() {
  # 0) archive the 50-code probe logs; the new probe writes the same filenames
  mkdir -p "$PROBES/50code"
  cp -f "$PROBES"/probe*.log "$PROBES"/PROBE_MAIN.log* "$PROBES/50code/" 2>/dev/null
  log "lane0: archived the 50-code probe logs to $PROBES/50code/"

  # 1) full-scale precision probe (4 runs x 3 epochs, serial on GPU0)
  rm -f /tmp/precision_probe.lock
  # Only P1/P2 run: cells 1 (all fp32) and 2 (all bf16) already serve as the references,
  # and --epochs 20 --max-steps 8925 keeps the learning-rate schedule identical, so the
  # first three epochs are directly comparable.
  log "lane0: starting the full-scale precision probe (P1/P2 only, about 10.5 h)"
  bash "$SCRIPTS/run_precision_probe.sh" --version mimic3 --epochs 20 --max-steps 8925 \
       --probes 2,3 --gpu 0 --launch --parallel >> "$LOG" 2>&1
  wait_rc "$PROBE_RC" 1200

  # 2) fill the window while GPU1 is busy: three seeds on one GPU
  log "lane0: mimic3-50, three seeds on one GPU"
  bash "$SCRIPTS/run_seeds.sh" --version mimic3-50 --cells 1,2,3 --seeds 1 --gpus 0 >> "$LOG" 2>&1

  # 3) speed re-measurement (non-fatal)
  log "lane0: speed re-measurement"
  bash "$SCRIPTS/run_speed_retest.sh" --gpu 0 >> "$LOG" 2>&1 || true

  touch "$GPU0_FREE"
  log "lane0 done: GPU0 released"
}

lane1() {
  log "lane1: waiting on cell 7 (${C7_RC})"
  wait_rc "$C7_RC" 1800
  log "lane1: waiting for GPU0 to be released (${GPU0_FREE})"
  wait_file "$GPU0_FREE" 1200
  log "lane1: mimic3-50, remaining seeds on two GPUs"
  bash "$SCRIPTS/run_seeds.sh" --version mimic3-50 --cells 1,2,3,5 --seeds 3 --gpus "0 1" >> "$LOG" 2>&1
  log "lane1: long-tail dump"
  mkdir -p "$OUT/longtail"
  python3 -u "$SCRIPTS/dump_predictions.py" --all --runs "$RUNS" \
      --out "$OUT/longtail" >> "$LOG" 2>&1 || log "[warn] long-tail dump failed (non-fatal)"
  touch "$DONE"
  log "lane1 done: pipeline finished"
}

if [ "${1:-}" = "--status" ]; then
  echo "=== pipeline v2 status ($(date '+%F %T')) ==="
  echo "probe rc    : $(cat "$PROBE_RC" 2>/dev/null || echo running/not started)"
  echo "cell 7 rc   : $(cat "$C7_RC" 2>/dev/null || echo running/not started)"
  echo "GPU0 free   : $([ -f "$GPU0_FREE" ] && echo yes || echo no)"
  echo "finished    : $([ -f "$DONE" ] && echo yes || echo no)"
  echo "GPU          : $(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | tr '\n' ' ')"
  echo "--- log tail ---"; tail -8 "$LOG" 2>/dev/null || echo "(none)"
  exit 0
fi

log "===== pipeline v2 started =====
rm -f "$GPU0_FREE" "$DONE"
lane0 & P0=$!
lane1 & P1=$!
wait $P0; R0=$?
wait $P1; R1=$?
log "===== pipeline v2 finished lane0 rc=$R0 lane1 rc=$R1 =====
touch "$DONE"
