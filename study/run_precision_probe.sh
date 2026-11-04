#!/bin/bash
# =============================================================================
# run_precision_probe.sh -- precision ablation: is the bf16 damage on the text side or the
# label side?
#
# bf16 costs the BiLSTM baseline several AUC-macro points while leaving the Transformer
# unharmed, and the mechanism is that rounding error accumulates along the recurrent path
# whereas each Transformer layer renormalises. This probe puts bf16 on **one side only**
# and thereby localises the damage:
#
#     P1  --bf16 --label_fp32   text bf16 / label fp32
#     P2  --bf16 --text_fp32    text fp32 / label bf16
#
#   P1 collapses and P2 is fine  -> the damage is on the text side
#   P2 collapses and P1 is fine  -> the damage is on the label side
#   both collapse                -> both contribute
#   reference: cell 1 (all fp32) and cell 2 (all bf16) from the same setting
#
# The 50-code setting is tried first because a 3-epoch run there takes well under an hour;
# if the effect shows up, the mechanism is settled cheaply. If not, rerun at full scale.
#
# Usage:
#   bash run_precision_probe.sh --version mimic3-50 --epochs 3 --launch   # P1/P2 on two GPUs
#   bash run_precision_probe.sh --now --probe 2 --gpu 0                    # one probe, foreground
# =============================================================================
set -u
VERSION=mimic3-50; EPOCHS=3; MAXSTEPS=0; PROBE=2; GPU=0; MODE=list; PARALLEL=0; PROBES=""
TOTALMAX=0; SUFFIX=""; RUNS_OVERRIDE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --version) VERSION="$2"; shift ;;
    --epochs)  EPOCHS="$2"; shift ;;
    --max-steps) MAXSTEPS="$2"; shift ;;
    --total-max-steps) TOTALMAX="$2"; shift ;;
    --suffix)  SUFFIX="$2"; shift ;;
    --runs)    RUNS_OVERRIDE="$2"; shift ;;
    --probe)   PROBE="$2"; shift ;;
    --probes)  PROBES="$2"; shift ;;
    --gpu)     GPU="$2"; shift ;;
    --parallel) PARALLEL=1 ;;
    --list)    MODE=list ;;
    --launch)  MODE=launch ;;
    --now)     MODE=now ;;
    *) echo "unknown argument $1"; exit 1 ;;
  esac
  shift
done

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${ICD_DATA_DIR:?set ICD_DATA_DIR to the prepared data directory}"
[ "$VERSION" = "mimic3-50" ] && DATA_DIR="${ICD_DATA_50:-$DATA_DIR/../mimic3_50}"
WE="${ICD_WORD_EMBEDDING:?set ICD_WORD_EMBEDDING to the word2vec model}"
REPO="${ICD_REPO:-$HERE/../model/hybrid}"
RES="${OUT_DIR:-$HERE/out/probes}"
RUNS="${RUNS_DIR:-$HERE/out/runs_probes}"
# --runs / --suffix: a diagnostic run must write to another directory or tag, otherwise it
# shares a run directory with a later probe and both append to the same epochs.jsonl.
[ -n "$RUNS_OVERRIDE" ] && RUNS="$RUNS_OVERRIDE"
SCRIPTS=$HERE/../bench

COMMON="--version $VERSION --decoder MultiLabelMultiHeadLAATV2 \
 --attention_dim 512 --truncate_length 4000 --batch_size 16 \
 --learning_rate 5e-4 --gradient_accumulation_steps 1 --xavier \
 --main_code_loss_weight 0.0 --rdrop_alpha 5.0 --est_cls 1 \
 --word_embedding_path $WE --train_epoch $EPOCHS --n_gpu 1 \
 --output_base_dir $RUNS --save_every 1 --keep_epochs 10 --log_every 50"
# --max-steps > 0 keeps --train_epoch, and therefore the learning-rate horizon, the same as
# the full run and only truncates early, so the first N epochs are directly comparable.
[ "${MAXSTEPS:-0}" -gt 0 ] && COMMON="$COMMON --max_steps $MAXSTEPS"
# --total-max-steps truncates globally across epochs; --max-steps compares a within-epoch
# index, so a value above steps/epoch never fires.
[ "${TOTALMAX:-0}" -gt 0 ] && COMMON="$COMMON --total_max_steps $TOTALMAX"
# Same encoder as cells 1/2 (BiLSTM 2x256); precision is the only change
BASE="--combiner lstm --rnn_dim 256 --num_layers 2 --attention_head 4 --term_count 4 --dec_chunk_codes 2048"

probe_args () {
  case "$1" in
    0) echo "$BASE" ;;
    1) echo "$BASE --bf16" ;;
    2) echo "$BASE --bf16 --label_fp32" ;;
    3) echo "$BASE --bf16 --text_fp32" ;;
  esac
}
probe_name () {
  case "$1" in
    0) echo "ref_all_fp32" ;; 1) echo "ref_all_bf16" ;;
    2) echo "P1_text_bf16_label_fp32" ;; 3) echo "P2_text_fp32_label_bf16" ;;
  esac
}

summarize () {   # print this probe's per-epoch numbers instead of digging through logs
  local name="$1"
  RUNS="$RUNS" python3 - "$name$SUFFIX" <<'PY' 2>/dev/null || true
import glob, json, os, sys
p = glob.glob(os.path.join(os.environ.get("RUNS", ""$RUNS""),
                           "*probe_%s" % sys.argv[1], "epochs.jsonl"))
if not p:
    print("   (no epochs.jsonl yet)"); raise SystemExit
for r in [json.loads(l) for l in open(p[0])]:
    print("   ep%d  testAUCma %.4f  F1mi %.4f  train_loss %.5f"
          % (r["epoch"], r["test"]["auc_macro"], r["test"]["f1_micro"], r["train_loss"]))
PY
}

echo "=== precision ablation plan ==="
echo "data: $VERSION ($DATA_DIR)   epochs: $EPOCHS   GPU: $GPU   code: $REPO"
for i in 0 1 2 3; do printf "  %-26s %s\n" "$(probe_name $i)" "$(probe_args $i)"; done
echo "reading: P1 collapses and P2 holds -> text side; the reverse -> label side; both -> both"
[ "$MODE" = "list" ] && exit 0

cd "$REPO"
export ICD_DATA_DIR="$DATA_DIR" ICD_WORD_EMBEDDING="$WE"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8

run_one () {   # run_one <probe> <GPU>
  local i="$1" gpu="$2" name; name="probe_$(probe_name "$i")$SUFFIX"
  local log="$RES/$name.log"
  echo "[probe] ===== $name -> GPU${gpu} ($(date '+%F %T')) ====="
  CUDA_VISIBLE_DEVICES=$gpu python -u main.py $COMMON $(probe_args "$i") --tag "$name" > "$log" 2>&1
  local rc=$?
  echo "$rc" > "$log.rc"
  tr '\r' '\n' < "$log" | grep -vE '^[[:space:]]*$' > "$log.clean.log" 2>/dev/null
  echo "[probe] $name done rc=$rc"; summarize "$(probe_name "$i")"
}

if [ "$MODE" = "now" ]; then
  mkdir -p "$RES" "$RUNS"
  run_one "$PROBE" "$GPU"
  exit $?
fi

# ── --launch: write a wrapper, then start detached ───────────────────────
#   --probes "0,1,2,3" runs those probes serially on one GPU
#   without --probes: two-GPU mode, P1 on GPU0 and P2 on GPU1
if [ "$PARALLEL" = 0 ]; then
  N=$(pgrep -f 'python -u main\.py' | wc -l | tr -d ' ')
  if [ "$N" != "0" ]; then
    echo "FAIL $N main.py processes already running; add --parallel to override"; exit 1
  fi
fi
LOCK=/tmp/precision_probe.lock
[ -f "$LOCK" ] && { echo "FAIL lock present (${LOCK}); remove it if nothing is running"; exit 1; }
mkdir -p "$RES" "$RUNS"

WRAP="$RES/probe_launch.run.sh"
# The wrapper re-invokes this script with --now, so any new argument that is not forwarded
# is silently dropped at that layer. New arguments must be added to EXTRA_ARGS too.
EXTRA_ARGS=""
[ -n "$RUNS_OVERRIDE" ] && EXTRA_ARGS="$EXTRA_ARGS --runs $RUNS_OVERRIDE"
[ -n "$SUFFIX" ] && EXTRA_ARGS="$EXTRA_ARGS --suffix $SUFFIX"
[ "${TOTALMAX:-0}" -gt 0 ] && EXTRA_ARGS="$EXTRA_ARGS --total-max-steps $TOTALMAX"
if [ -n "$PROBES" ]; then
  { echo '#!/bin/bash'; echo 'set -u'
    echo ""
    echo "cd $REPO"
    echo "export ICD_DATA_DIR=$DATA_DIR ICD_WORD_EMBEDDING=$WE"
    echo "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8"
    echo "rcall=0"
    IFS=',' ; for i in $PROBES; do
      echo "$SCRIPTS/run_precision_probe.sh --version $VERSION --epochs $EPOCHS --max-steps $MAXSTEPS$EXTRA_ARGS --now --probe $i --gpu $GPU > $RES/probe${i}.log 2>&1 || rcall=1"
    done; unset IFS
    echo "echo \"serial probes [$PROBES] all done rcall=\${rcall} (\$(date '+%F %T'))\" >> $RES/PROBE_MAIN.log"
    echo "rm -f $LOCK"
  } > "$WRAP"
  echo "[probe] serial mode: probes [$PROBES] in turn on GPU${GPU}; logs $RES/probe<N>.log"
else
  cat > "$WRAP" <<WRAPEOF
#!/bin/bash
set -u
cd "$REPO"
export ICD_DATA_DIR="$DATA_DIR" ICD_WORD_EMBEDDING="$WE"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
$SCRIPTS/run_precision_probe.sh --version $VERSION --epochs $EPOCHS --max-steps $MAXSTEPS$EXTRA_ARGS --now --probe 2 --gpu 0 > $RES/P1.log 2>&1 &
p1=\$!
$SCRIPTS/run_precision_probe.sh --version $VERSION --epochs $EPOCHS --max-steps $MAXSTEPS$EXTRA_ARGS --now --probe 3 --gpu 1 > $RES/P2.log 2>&1 &
p2=\$!
wait \$p1 \$p2
rc=\$?
echo "P1/P2 all done rc=\${rc} ($(date '+%F %T'))" >> $RES/PROBE_MAIN.log
rm -f $LOCK
WRAPEOF
  echo "[probe] two-GPU mode: P1 on GPU0, P2 on GPU1"
fi
chmod +x "$WRAP"
bash "$SCRIPTS/launch_detached.sh" "$RES/PROBE_MAIN.log" "$LOCK" "bash $WRAP"
echo "[probe] launched detached; follow with: tail -20 $RES/probe*.log"
