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
#   cell 7  hybrid, label bf16         3->7  separate BiLSTM on the label branch, text on FA
#   cell 8  hybrid, label fp32         7->8  pin the label branch to fp32 (the repair)
# Cells 7 and 8 run from model/hybrid; the others from model/main.
#
# Usage:
#   bash run_matrix_v2.sh --dry        # 5 steps per cell, no evaluation: real s/step and peak
#   bash run_matrix_v2.sh              # full run, --epochs 20 (Table 3 of the paper)
#   bash run_matrix_v2.sh --cells 3,6  # a subset
#   bash run_matrix_v2.sh --epochs 1   # one epoch per cell, for a plumbing check
#   bash run_matrix_v2.sh --1gpu       # single device (effective batch halves)
#   bash run_matrix_v2.sh --steps 300  # convergence pre-check over 300 steps
#
# Every cell is launched with --tag, so its run directory ends in the cell name (e.g.
# word_...-_1_msmn_fp32_naive). analysis/ and figures/ match run directories on that
# suffix; without --tag none of the numbers are reachable for them. The mapping this
# script produced is also written to $OUT/INDEX.tsv.
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
REPO_MAIN="${ICD_REPO:-$HERE/../model/main}"
REPO_HYBRID="${ICD_REPO_HYBRID:-$HERE/../model/hybrid}"
# Run directories are an output of the study, not scratch: default them inside the study
# tree, so analysis/ and figures/ find them without a copy step.
OUT="${OUT_DIR:-$HERE/out/runs_v2}"
DRY=0
LIVE=1
CELLS="1,2,3,4,5,6,7,8"
EPOCHS=20                     # Table 3 of the paper
STEPS=0                       # >0: run only N steps (convergence pre-check)
NPROC=2
TIMEOUT=0                     # >0: seconds per cell; 0 leaves the run uncapped
PRINT_ARGS=0                  # 1: print each cell's command and exit (no GPU touched)

while [ $# -gt 0 ]; do
  case "$1" in
    --dry)   DRY=1 ;;
    --cells) CELLS="$2"; shift ;;
    --epochs) EPOCHS="$2"; shift ;;
    --steps) STEPS="$2"; shift ;;
    --1gpu)  NPROC=1 ;;
    --log-only) LIVE=0 ;;
    --print-args) PRINT_ARGS=1 ;;
    *) echo "unknown argument $1"; exit 1 ;;
  esac
  shift
done

mkdir -p "$OUT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

# Shared arguments, as in the MSMN paper's Table 2 and this paper's Table 3
COMMON="--version mimic3 --decoder MultiLabelMultiHeadLAATV2 \
 --attention_dim 512 --truncate_length 4000 --batch_size 8 \
 --learning_rate 5e-4 --gradient_accumulation_steps 1 --xavier \
 --main_code_loss_weight 0.0 --rdrop_alpha 5.0 --est_cls 1 \
 --word_embedding_path $WE --output_base_dir $OUT"
# Label branch: the two padding-free representations, with chunked checkpointing
#   bucketed = dense SDPA, explicitly non-flash via --enc_attn_backend efficient
#   varlen   = flash-attn variable-length (cell 5, where the encoder also uses flash)
BUCKETED="--label_repr bucketed --enc_chunk_tokens 16384 --enc_attn_backend efficient"

# Which model tree a cell runs from: cells 7/8 carry the hybrid label branch, which only
# exists in model/hybrid.
cell_repo () {
  case "$1" in
    7|8) echo "$REPO_HYBRID" ;;
    *)   echo "$REPO_MAIN" ;;
  esac
}
cell_args () {
  case "$1" in
    1) echo "--combiner lstm --rnn_dim 256 --num_layers 2 --attention_head 4 --term_count 4 --dec_chunk_codes 2048" ;;
    2) echo "--combiner lstm --rnn_dim 256 --num_layers 2 --attention_head 4 --term_count 4 --dec_chunk_codes 2048 --bf16" ;;
    3) echo "--combiner transformer --rnn_dim 512 --num_transformer_layers 4 --pos_embed learn --attention_head 4 --term_count 4 $BUCKETED --dec_chunk_codes 2048 --bf16" ;;
    4) echo "--combiner transformer --rnn_dim 512 --num_transformer_layers 4 --pos_embed learn --attention_head 4 --term_count 4 $BUCKETED --attn_impl mem_efficient --bf16" ;;
    5) echo "--combiner transformer --rnn_dim 512 --num_transformer_layers 4 --pos_embed learn --attention_head 4 --term_count 4 --label_repr varlen --enc_chunk_tokens 16384 --attn_impl varlen --bf16" ;;
    6) echo "--combiner transformer --rnn_dim 512 --num_transformer_layers 4 --pos_embed learn --attention_head 4 --term_count 4 $BUCKETED --dec_chunk_codes 2048" ;;
    # 7: text branch Transformer + flash, label branch on its own BiLSTM (padded, forced).
    #    --label_repr is deliberately absent: --label_combiner lstm forces that branch to
    #    padded, and create_label_config follows the branch, not the flag.
    7) echo "--combiner transformer --rnn_dim 512 --num_transformer_layers 4 --pos_embed learn --attention_head 4 --term_count 4 --label_combiner lstm --label_rnn_dim 256 --label_lstm_layers 2 --attn_impl varlen --bf16" ;;
    # 8: the wiring of 7 with the label branch pinned to fp32. --label_fp32 cannot be
    #    combined with label_repr varlen, and the LSTM branch is padded anyway, so the
    #    label branch uses the bucketed representation.
    8) echo "--combiner transformer --rnn_dim 512 --num_transformer_layers 4 --pos_embed learn --attention_head 4 --term_count 4 --label_combiner lstm --label_rnn_dim 256 --label_lstm_layers 2 $BUCKETED --dec_chunk_codes 2048 --bf16 --label_fp32" ;;
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
    7) echo "7_hybrid_lstm_label_fa_text" ;;
    8) echo "8_hybrid_labelfp32" ;;
  esac
}
# The run directory this cell wrote: the newest directory whose name ends in the tag.
cell_dir () {
  local name="$1"
  local d
  d=$(ls -td "$OUT"/*"$name" 2>/dev/null | head -1)
  echo "${d:-}"
}

run_cell () {
  local i="$1"
  local name; name=$(cell_name "$i")
  if [ -z "$name" ]; then
    echo "unknown cell $i (known: 1-8)"; return 1
  fi
  local args; args=$(cell_args "$i")
  local repo; repo=$(cell_repo "$i")
  local log="$OUT/${name}.log"
  local maxsteps=""
  [ "$DRY" = "1" ] && maxsteps="--max_steps 5 --skip_eval"
  [ "$STEPS" -gt 0 ] && maxsteps="--max_steps $STEPS --skip_eval"

  echo "########## cell $i : $name ##########"
  echo "  repo=$repo epochs=$EPOCHS"
  local t0=$SECONDS
  # No wall-clock cap by default: at 20 epochs a full-scale cell needs up to ~72 h on one
  # device, so a fixed timeout would kill exactly the runs this reproduces. Set TIMEOUT to
  # bound it (seconds).
  local guard=""
  [ "${TIMEOUT:-0}" -gt 0 ] && guard="timeout $TIMEOUT"
  # --tag is what makes the run directory findable by analysis/ and figures/: they match on
  # the name suffix, and generate_output_folder_name() only appends it when asked.
  if [ "$NPROC" = "1" ]; then
    CMD="$guard python -u main.py $COMMON $args --train_epoch $EPOCHS $maxsteps --n_gpu 1 --tag $name"
  else
    CMD="$guard accelerate launch --num_processes $NPROC --main_process_port 29518 \
        main.py $COMMON $args --train_epoch $EPOCHS $maxsteps --n_gpu $NPROC --tag $name"
  fi
  local rc
  if [ "$LIVE" = "1" ]; then
    ( cd "$repo" && eval "$CMD" ) 2>&1 | tee "$log"
    rc=${PIPESTATUS[0]}
  else
    ( cd "$repo" && eval "$CMD" ) > "$log" 2>&1
    rc=$?
  fi
  echo "$rc" > "$log.rc"
  tr '\r' '\n' < "$log" | grep -vE '^\s*$' > "${log%.log}.clean.log" 2>/dev/null || true
  local dt=$((SECONDS - t0))

  # Pull peak memory and per-step time out of the log (--dry gets them from [max_steps])
  local mem; mem=$(grep -oE "peak allocated=[0-9.]+ GB reserved=[0-9.]+ GB" "$log" | tail -1)
  local spm; spm=$(grep -oE "step [0-9.]+ ms" "$log" | tail -1)
  local loss; loss=$(grep -oE "Epoch: [0-9.]+" "$log" | tail -1 | cut -d' ' -f2)
  local dir; dir=$(cell_dir "$name")
  printf "  rc=%s elapsed=%ss loss=%s\n" "$rc" "$dt" "${loss:-?}"
  [ -n "$mem" ] && echo "  $mem | $spm"
  if [ -n "$dir" ]; then
    echo "  run dir: $dir"
  else
    echo "  [warn] no run directory matching *$name under $OUT"
  fi
  if [ $rc -ne 0 ]; then
    echo "  [warn] failed; last 6 lines:"
    tail -6 "$log" | sed 's/^/     /'
  fi
  echo "$name|rc=$rc|${dt}s|loss=${loss:-?}|${mem:-?}|${spm:-?}" >> "$OUT/SUMMARY.txt"
  printf "%s\t%s\t%s\n" "$name" "$repo" "${dir:-MISSING}" >> "$OUT/INDEX.tsv"
  echo
}

# --print-args: emit "<cell>\t<repo>\t<args>" and stop. It exists so the flags can be
# validated against the real parser without a GPU (tests/check_cell_args.sh does exactly
# that): a cell whose flags drifted out of the parser costs hours to discover otherwise.
if [ "$PRINT_ARGS" = "1" ]; then
  for i in $(echo "$CELLS" | tr ',' ' '); do
    printf "%s\t%s\t%s\t%s\n" "$i" "$(cell_name "$i")" "$(cell_repo "$i")" "$(cell_args "$i")"
  done
  exit 0
fi

: > "$OUT/SUMMARY.txt"
: > "$OUT/INDEX.tsv"
echo "mode: $([ "$DRY" = "1" ] && echo '--dry (5 steps, no evaluation)' || echo "full (${EPOCHS} epochs)") | processes=$NPROC" | tee -a "$OUT/SUMMARY.txt"
echo "run root: $OUT" | tee -a "$OUT/SUMMARY.txt"
for i in $(echo "$CELLS" | tr ',' ' '); do
  run_cell "$i"
done

echo "================ summary ================"
column -t -s'|' "$OUT/SUMMARY.txt" 2>/dev/null || cat "$OUT/SUMMARY.txt"
echo
echo "run dirs: $OUT  (index: $OUT/INDEX.tsv)"
echo "next:"
echo "  export ICD_RESULTS_ROOT=$(cd "$OUT/.." && pwd)"
echo "  python analysis/summarize_results.py --runs $OUT --out $HERE/out/summary"
echo "  python figures/make_figures.py --root $(cd "$OUT/.." && pwd)"
