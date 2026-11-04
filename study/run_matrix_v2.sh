#!/bin/bash
# =============================================================================
# run_matrix_v2.sh -- the eight-cell study, cell by cell (real MIMIC-III, 2x RTX 4090)
#
# Compared with the earlier matrix, one thing had to change: feeding the label branch in
# the padded representation puts 1,070,520 slots behind 200,353 real tokens (5.34x), and
# in fp32 that does not fit in 24 GB. Every cell here therefore uses --label_repr bucketed
# on the label branch: exact length grouping, no padding waste, no mask, and fp32 works.
# The fastest bf16 route remains varlen with flash-attn.
#
# Consecutive cells differ in exactly one variable:
#   cell 1  MSMN fp32 naive            reference baseline
#   cell 2  MSMN bf16 naive            1->2  precision, on the baseline
#   cell 3  Transf bf16 naive          2->3  encoder architecture at fixed precision
#   cell 4  Transf bf16 mem_efficient  3->4  decoder implementation
#   cell 5  Transf bf16 varlen+FA      4->5  all three attention sites onto flash
#   cell 6  Transf fp32 naive          3->6  precision, on the Transformer
#
# Usage:
#   bash run_matrix_v2.sh --dry        # 5 steps per cell, no evaluation: real s/step and peak
#   bash run_matrix_v2.sh              # full run (1 epoch per cell)
#   bash run_matrix_v2.sh --cells 3,6  # a subset
#   bash run_matrix_v2.sh --1gpu       # single device (effective batch halves)
#   bash run_matrix_v2.sh --steps 300  # convergence pre-check over 300 steps
#
# Requires:
#   export ICD_DATA_DIR=/path/to/mimic3_caml
#   export ICD_WORD_EMBEDDING=/path/to/word2vec_sg0_100.model
# Long runs belong in tmux.
# =============================================================================
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATA_DIR="${ICD_DATA_DIR:?set ICD_DATA_DIR to the prepared data directory}"
WE="${ICD_WORD_EMBEDDING:?set ICD_WORD_EMBEDDING to the word2vec model}"
REPO="${ICD_REPO:-$HERE/../model/main}"
OUT="${OUT_DIR:-$HERE/out/matrix_v2}"
DRY=0
LIVE=1
CELLS="1,2,3,4,5,6"
STEPS=0                       # >0: run only N steps (convergence pre-check)
NPROC=2

while [ $# -gt 0 ]; do
  case "$1" in
    --dry)   DRY=1 ;;
    --cells) CELLS="$2"; shift ;;
    --steps) STEPS="$2"; shift ;;
    --1gpu)  NPROC=1 ;;
    --log-only) LIVE=0 ;;
    *) echo "unknown argument $1"; exit 1 ;;
  esac
  shift
done

cd "$REPO" || exit 1
mkdir -p "$OUT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

# Shared arguments, as in the MSMN paper's Table 2 and this paper's Table 3
COMMON="--version mimic3 --decoder MultiLabelMultiHeadLAATV2 \
 --attention_dim 512 --truncate_length 4000 --batch_size 8 \
 --learning_rate 5e-4 --gradient_accumulation_steps 1 --xavier \
 --main_code_loss_weight 0.0 --rdrop_alpha 5.0 --est_cls 1 \
 --word_embedding_path $WE --output_base_dir /tmp/matrix_v2_out"
# Label branch: the two padding-free representations, with chunked checkpointing
#   bucketed = dense SDPA, explicitly non-flash via --enc_attn_backend efficient
#   varlen   = flash-attn variable-length (cell 5, where the encoder also uses flash)
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
    1) echo "1_msmn_fp32_naive" ;;
    2) echo "2_msmn_bf16_naive" ;;
    3) echo "3_transf_bf16_naive" ;;
    4) echo "4_transf_bf16_memeff" ;;
    5) echo "5_transf_bf16_varlen_fa" ;;
    6) echo "6_transf_fp32_naive" ;;
  esac
}

run_cell () {
  local i="$1"
  local name; name=$(cell_name "$i")
  local args; args=$(cell_args "$i")
  local log="$OUT/${name}.log"
  local epochs=1 maxsteps=""
  [ "$DRY" = "1" ] && maxsteps="--max_steps 5 --skip_eval"
  [ "$STEPS" -gt 0 ] && maxsteps="--max_steps $STEPS --skip_eval"

  echo "########## cell $i : $name ##########"
  local t0=$SECONDS
  if [ "$NPROC" = "1" ]; then
    CMD="timeout 14400 python -u main.py $COMMON $args --train_epoch $epochs $maxsteps --n_gpu 1"
  else
    CMD="timeout 14400 accelerate launch --num_processes $NPROC --main_process_port 29518 \
        main.py $COMMON $args --train_epoch $epochs $maxsteps --n_gpu $NPROC"
  fi
  if [ "$LIVE" = "1" ]; then
    eval "$CMD" 2>&1 | tee "$log"
    local rc=${PIPESTATUS[0]}
  else
    eval "$CMD" > "$log" 2>&1
    local rc=$?
  fi
  tr '\r' '\n' < "$log" | grep -vE '^\s*$' > "${log%.log}.clean.log" 2>/dev/null || true
  local dt=$((SECONDS - t0))

  # Pull peak memory and per-step time out of the log (--dry gets them from [max_steps])
  local mem; mem=$(grep -oE "peak allocated=[0-9.]+ GB reserved=[0-9.]+ GB" "$log" | tail -1)
  local spm; spm=$(grep -oE "step [0-9.]+ ms" "$log" | tail -1)
  local loss; loss=$(grep -oE "Epoch: [0-9.]+" "$log" | tail -1 | cut -d' ' -f2)
  printf "  rc=%s elapsed=%ss loss=%s\n" "$rc" "$dt" "${loss:-?}"
  [ -n "$mem" ] && echo "  $mem | $spm"
  if [ $rc -ne 0 ]; then
    echo "  [warn] failed; last 6 lines:"
    tail -6 "$log" | sed 's/^/     /'
  fi
  echo "$name|rc=$rc|${dt}s|loss=${loss:-?}|${mem:-?}|${spm:-?}" >> "$OUT/SUMMARY.txt"
  echo
}

: > "$OUT/SUMMARY.txt"
echo "mode: $([ "$DRY" = "1" ] && echo '--dry (5 steps, no evaluation)' || echo 'full') | processes=$NPROC" | tee -a "$OUT/SUMMARY.txt"
for i in $(echo "$CELLS" | tr ',' ' '); do
  run_cell "$i"
done

echo "================ summary ================"
column -t -s'|' "$OUT/SUMMARY.txt" 2>/dev/null || cat "$OUT/SUMMARY.txt"
echo
echo "logs: $OUT"
