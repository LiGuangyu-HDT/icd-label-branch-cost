#!/bin/bash
# =============================================================================
# verify_b16.sh -- check the "one cell per GPU, batch 16 per GPU" mode
#
# Why this needs checking: the validated mode is one cell on two GPUs with DDP and
# batch 8 per GPU. Running two cells side by side, each on one GPU with batch 16, halves
# wall-clock:
#   * the effective batch is 16 either way (train_utils.py defines bsz that way) and the
#     loss is a mean, so the two arrangements are mathematically equivalent;
#   * the label branch (35,684 descriptions) is recomputed on every device, so DDP does
#     not save it -- two cells in parallel therefore give twice the throughput.
# The risks are memory at batch 16 on one GPU, and the two jobs slowing each other down.
#
# Usage:
#   bash verify_b16.sh 3 6 100      # cell 3 on GPU0, cell 6 on GPU1, 100 steps each
#
# cell_args / cell_name must stay in step with run_matrix_v2.sh.
# =============================================================================
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CELL_A="${1:-3}"; CELL_B="${2:-6}"; STEPS="${3:-100}"
REPO="${ICD_REPO:-$HERE/../model/main}"
OUT="${OUT_DIR:-$HERE/out}"
RUNS="${RUNS_DIR:-$HERE/out/runs_b16}"
BS=16

: "${ICD_DATA_DIR:?set ICD_DATA_DIR to the prepared data directory}"
: "${ICD_WORD_EMBEDDING:?set ICD_WORD_EMBEDDING to the word2vec model}"
cd "$REPO" || exit 1
mkdir -p "$OUT" "$RUNS"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8

COMMON="--version mimic3 --decoder MultiLabelMultiHeadLAATV2 \
 --attention_dim 512 --truncate_length 4000 --batch_size $BS \
 --learning_rate 5e-4 --gradient_accumulation_steps 1 --xavier \
 --main_code_loss_weight 0.0 --rdrop_alpha 5.0 --est_cls 1 \
 --word_embedding_path $WE --train_epoch 1 --max_steps $STEPS --skip_eval --n_gpu 1 \
 --output_base_dir $RUNS"
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
    1) echo "c1_msmn_fp32" ;; 2) echo "c2_msmn_bf16" ;;
    3) echo "c3_tf_bf16_naive" ;; 4) echo "c4_tf_bf16_memeff" ;;
    5) echo "c5_tf_bf16_fa" ;; 6) echo "c6_tf_fp32_naive" ;;
  esac
}

run_one () {                    # run_one <cell> <gpu> <tag>
  local c="$1" gpu="$2" tag="$3"
  local log="$OUT/9D_b16_${WAVE:-w}_${tag}.log"
  CUDA_VISIBLE_DEVICES=$gpu python -u main.py $COMMON $(cell_args "$c") \
      --tag "b16g${gpu}" > "$log" 2>&1
  echo $? > "${log}.rc"
}

NA=$(cell_name "$CELL_A"); NB=$(cell_name "$CELL_B")
WAVE="${WAVE:-$(date +%H%M)}"
export WAVE
echo "[verify] cell $CELL_A ($NA) -> GPU0 ; cell $CELL_B ($NB) -> GPU1 ; $STEPS steps each ; batch $BS on one GPU"
echo "[verify] effective batch = $BS x 1 GPU = ${BS}, which is the paper's 16"

run_one "$CELL_A" 0 "$NA" &
PA=$!
run_one "$CELL_B" 1 "$NB" &
PB=$!
wait $PA $PB

echo
echo "================ results ================"
for t in "$NA" "$NB"; do
  L="$OUT/9D_b16_${t}.log"
  rc=$(cat "${L}.rc" 2>/dev/null || echo "?")
  mem=$(grep -oE "peak allocated=[0-9.]+ GB" "$L" | tail -1)
  step=$(grep -oE "step [0-9.]+ ms" "$L" | tail -1)
  l1=$(grep -oE "\[loss\] step=[0-9]+ \| step [0-9.]+" "$L" | head -1)
  l2=$(grep -oE "\[loss\] step=[0-9]+ \| step [0-9.]+" "$L" | tail -1)
  fat=$(grep -c "FATAL" "$L")
  oom=$(grep -ci "out of memory" "$L")
  printf "%-18s rc=%-3s %s | %s | FATAL=%s OOM=%s\n  first:%s  last:%s\n" \
     "$t" "$rc" "${mem:-none}" "${step:-none}" "$fat" "$oom" "$l1" "$l2"
done
df -h "$OUT" | tail -1
