#!/bin/bash
# =============================================================================
# run_cell7_rerun.sh -- cell 7 replication and the normalisation probe
#
# Cell 7 (hybrid, label branch bf16) collapses to chance. Two things in the paper rest on
# re-running it, and this script produces both:
#
#   Phase A  three independent re-runs of cell 7, to replace the single original run with a
#            mean and a spread, and to supply the per-bucket profile the original run's
#            checkpoint could not provide.
#   Phase B  one run with --label_layer_norm, which puts a LayerNorm on the BiLSTM's output.
#            This tests prediction (i) of the mechanism account in section 7 of the paper:
#            "adding a normalisation layer to the recurrent label branch should recover a
#            substantial part of the loss". It does not, and the negative result is reported.
#
# Why --label_layer_norm and not --layer_norm: --layer_norm also changes the text branch, so
# it would not isolate the label branch's output scale. The probe only varies one thing.
#
# Cost on one RTX 4090: ~52 min/epoch with three runs in parallel, ~17 min/epoch alone, so
# about 17 h for phase A and 6 h for phase B.
#
# The checkpoints were produced before --label_layer_norm existed, which is also why
# LSTM_Combiner reads its norm module through getattr: see the note in model/combiner.py.
#
# Usage:  ./run_cell7_rerun.sh [phase-a|phase-b|all]     (default: all)
# =============================================================================
set -eu

PHASE="${1:-all}"

: "${ICD_DATA_DIR:?set ICD_DATA_DIR to the directory holding mimic3/, D_ICD_DIAGNOSES.csv ...}"
: "${ICD_WORD_EMBEDDING:?set ICD_WORD_EMBEDDING to the word2vec .model file}"
REPO="${ICD_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
HYBRID="$REPO/model/hybrid"
RUNS="${ICD_RUNS_DIR:-$REPO/runs}"
GPUS="${ICD_GPUS:-1}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# One variable compared against cell 7 as published: only --label_layer_norm differs.
COMMON="--version mimic3 --decoder MultiLabelMultiHeadLAATV2 \
 --combiner transformer --rnn_dim 512 --num_transformer_layers 4 \
 --attention_head 4 --term_count 4 --label_combiner lstm --label_repr padded \
 --enc_chunk_tokens 16384 --attn_impl varlen --bf16 --dec_chunk_codes 0 \
 --batch_size 16 --train_epoch 20 --word_embedding_path $ICD_WORD_EMBEDDING \
 --xavier --main_code_loss_weight 0.0 --rdrop_alpha 5.0 --est_cls 1 \
 --learning_rate 5e-4 --weight_decay 0.01 --truncate_length 4000 \
 --n_gpu 1 --output_base_dir $RUNS"

launch() {   # launch <tag> <extra args...>
    local tag=$1; shift
    if compgen -G "$RUNS/*${tag}" > /dev/null; then
        echo "[skip] $tag already has a run directory under $RUNS" >&2
        return 0
    fi
    echo "[launch] $tag"
    ( cd "$HYBRID" && python -u main.py $COMMON "$@" --tag "$tag" )
}

run_phase_a() {
    echo "== phase A: three re-runs of cell 7, GPUS=$GPUS =="
    local tags=(cell7_rerun_s1 cell7_rerun_s2 cell7_rerun_s3)
    local i=0
    for t in "${tags[@]}"; do
        launch "$t" &
        i=$((i + 1))
        if [ "$i" -ge "$GPUS" ]; then wait; i=0; fi
    done
    wait
}

run_phase_b() {
    echo "== phase B: the normalisation probe =="
    launch cell7_layernorm --label_layer_norm
}

case "$PHASE" in
    phase-a) run_phase_a ;;
    phase-b) run_phase_b ;;
    all)     run_phase_a; run_phase_b ;;
    *) echo "usage: $0 [phase-a|phase-b|all]" >&2; exit 2 ;;
esac

# Long-tail export, once the runs above have finished:
#   python "$REPO/analysis/dump_predictions.py" --run-dir "$RUNS/<dir>" \
#          --repo "$HYBRID" --out "$RUNS/longtail/<tag>"
#   python "$REPO/analysis/bucket_metrics.py" \
#          --model <tag> <out>/y_true.npy <out>/y_prob.npy <out>/y_pred.npy \
#          --train-freq <out>/train_freq.npy --scheme 5buckets --out rows.tex
echo "[done] $PHASE"
