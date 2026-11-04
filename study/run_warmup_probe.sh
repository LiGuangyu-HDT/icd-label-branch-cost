#!/bin/bash
# =============================================================================
# run_warmup_probe.sh -- rule out "the Transformer only lost because you did not tune it"
#
# Cell 3 (Transformer, bf16) starts epochs 1-3 from a poor dev position, with a first-epoch
# training loss an order of magnitude above the BiLSTM's. That is exactly where a reviewer
# would challenge "cell 1 > cell 3", so this probe runs two configurations that change one
# knob each and compares their first three epochs against cell 3:
#
#   A  --warmup_ratio 0.06     warm-up only, learning rate unchanged
#   B  --learning_rate 2e-4    lower learning rate only, no warm-up
#   reference: cell 3's first three epochs
#
# The schedule must stay --train_epoch 20 with --max_steps 8925 (= 3 x 2975) truncating it,
# otherwise the linear decay differs and the comparison is meaningless.
#
# Usage:
#   bash run_warmup_probe.sh --gpu 0 --launch        # both configurations, serially
#   bash run_warmup_probe.sh --gpu 0 --now --cfg A   # one configuration, in the foreground
# =============================================================================
set -u
MODE=list; GPU=0; CFG=all; PARALLEL=0
while [ $# -gt 0 ]; do
  case "$1" in
    --gpu) GPU="$2"; shift ;;
    --cfg) CFG="$2"; shift ;;
    --parallel) PARALLEL=1 ;;
    --list) MODE=list ;;
    --launch) MODE=launch ;;
    --now) MODE=now ;;
    *) echo "unknown argument $1"; exit 1 ;;
  esac
  shift
done

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${ICD_DATA_DIR:?set ICD_DATA_DIR to the prepared data directory}"
WE="${ICD_WORD_EMBEDDING:?set ICD_WORD_EMBEDDING to the word2vec model}"
REPO="${ICD_REPO:-$HERE/../model/hybrid}"
RES="${OUT_DIR:-$HERE/out/probes}"
RUNS="${RUNS_DIR:-$HERE/out/runs_probes}"
SCRIPTS=$HERE/../bench
LOCK=/tmp/warmup_probe.lock
MAX_STEPS=8925            # 3 x 2975: keep the 20-epoch schedule, take only 3 epochs

COMMON="--version mimic3 --decoder MultiLabelMultiHeadLAATV2 \
 --attention_dim 512 --truncate_length 4000 --batch_size 16 \
 --learning_rate 5e-4 --gradient_accumulation_steps 1 --xavier \
 --main_code_loss_weight 0.0 --rdrop_alpha 5.0 --est_cls 1 \
 --word_embedding_path $WE --train_epoch 20 --n_gpu 1 \
 --output_base_dir $RUNS --save_every 1 --keep_epochs 10 --log_every 50 \
 --combiner transformer --rnn_dim 512 --num_transformer_layers 4 --pos_embed learn \
 --attention_head 4 --term_count 4 --label_repr bucketed --enc_chunk_tokens 16384 \
 --enc_attn_backend efficient --dec_chunk_codes 2048 --bf16 --max_steps $MAX_STEPS"

cfg_extra () {   # the only difference between A and B
  case "$1" in
    A) echo "--warmup_ratio 0.06" ;;
    B) echo "--learning_rate 2e-4" ;;
  esac
}
cfg_name () { case "$1" in A) echo "warmup_probe_A_warm006";; B) echo "warmup_probe_B_lr2e4";; esac; }

echo "=== warmup probe plan (GPU${GPU}, $MAX_STEPS steps = 3 epochs) ==="
for c in A B; do printf "  %-26s %s\n" "$(cfg_name $c)" "$(cfg_extra $c)"; done
echo "reference: cell 3 dev AUC-macro 0.7237 / 0.7825 / 0.8246 over its first three epochs"
[ "$MODE" = "list" ] && exit 0

cd "$REPO"
export ICD_DATA_DIR="$DATA_DIR" ICD_WORD_EMBEDDING="$WE"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
mkdir -p "$RES" "$RUNS"

run_cfg () {
  local c="$1" gpu="$2" name log rc
  name="$(cfg_name $c)"; log="$RES/$name.log"
  echo "[warmup] ===== $name -> GPU${gpu} ($(date '+%F %T')) ====="
  CUDA_VISIBLE_DEVICES=$gpu python -u main.py $COMMON $(cfg_extra "$c") --tag "$name" > "$log" 2>&1
  rc=$?
  echo "$rc" > "$log.rc"
  tr '\r' '\n' < "$log" | grep -vE '^[[:space:]]*$' > "$log.clean.log" 2>/dev/null
  echo "[warmup] $name done rc=$rc"
  python3 - "$name" <<'PY' 2>/dev/null || true
import glob, json, sys
p = glob.glob(""$RUNS"/*%s/epochs.jsonl" % sys.argv[1])
if p:
    for r in [json.loads(l) for l in open(p[0])]:
        print("   ep%d  dev AUCma %.4f  test AUCma %.4f  F1mi %.4f"
              % (r["epoch"], r["dev"]["auc_macro"], r["test"]["auc_macro"], r["test"]["f1_micro"]))
PY
}

if [ "$MODE" = "now" ]; then
  run_cfg "$CFG" "$GPU"; exit $?
fi

if [ "$PARALLEL" = 0 ]; then
  N=$(pgrep -f 'python -u main\.py' | wc -l | tr -d ' ')
  [ "$N" != "0" ] && { echo "FAIL $N main.py processes already running; add --parallel to override"; exit 1; }
fi
# Do not touch the lock here: the launcher checks and creates the same lock, and creating
# it first makes the launcher refuse to start, believing a run is already in progress.
if [ -f "$LOCK" ]; then
  # The lock is stale when no related python process is running; clear it then
  if ! pgrep -f "warmup_probe.cfg" >/dev/null 2>&1 && ! pgrep -f "warmup_launch.run" >/dev/null 2>&1; then
    echo "[warmup] stale lock found (no matching process), removed"; rm -f "$LOCK"
  else
    echo "FAIL lock present and a process appears to be running (${LOCK})"; exit 1
  fi
fi
WRAP="$RES/warmup_launch.run.sh"
cat > "$WRAP" <<WRAPEOF
#!/bin/bash
set -u
cd "$REPO"
export ICD_DATA_DIR="$DATA_DIR" ICD_WORD_EMBEDDING="$WE"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
$SCRIPTS/run_warmup_probe.sh --now --cfg A --gpu $GPU > $RES/warmupA.log 2>&1
$SCRIPTS/run_warmup_probe.sh --now --cfg B --gpu $GPU > $RES/warmupB.log 2>&1
echo "warmup A/B done (\$(date '+%F %T'))" >> $RES/PROBE_MAIN.log
rm -f $LOCK
WRAPEOF
chmod +x "$WRAP"
bash "$SCRIPTS/launch_detached.sh" "$RES/WARMUP_MAIN.log" "$LOCK" "bash $WRAP"
echo "[warmup] A/B launched detached on GPU${GPU}, serial, about 11.6 h"
