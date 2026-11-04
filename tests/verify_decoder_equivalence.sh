#!/bin/bash
# =============================================================================
# verify_decoder_equivalence.sh -- Table 2, V4: chunking the code axis is exact
#
# Chunking the 8,921-way decoder along the code axis is only legitimate if it does not
# change the arithmetic. This compares the chunked and unchunked decoder on *identical*
# weights, so any difference can only come from the chunking:
#
#   1) fix the weights with --save-weights
#   2) run unchunked   (--dec-chunk-codes 0)
#   3) run chunked     (--dec-chunk-codes 1024)
#   4) the logits must be bit-identical
#
# Fixed weights are used rather than a fixed seed because np.random is unseeded in this
# repository (the UNK/MASK embeddings and coder_init use it), so seeding torch alone does
# not reproduce initialisation.
#
# Usage:
#   export ICD_DATA_DIR=/path/to/mimic3_caml
#   export ICD_WORD_EMBEDDING=/path/to/word2vec_sg0_100.model
#   bash verify_decoder_equivalence.sh
# =============================================================================
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${ICD_REPO:-$HERE/../model/main}"
OUT="${OUT_DIR:-$HERE/out}"
W="$OUT/w_ref.pt"
mkdir -p "$OUT"

: "${ICD_DATA_DIR:?set ICD_DATA_DIR to the prepared data directory}"
: "${ICD_WORD_EMBEDDING:?set ICD_WORD_EMBEDDING to the word2vec model}"
cd "$REPO" || exit 1

LOG="$OUT/decoder_equivalence.jsonl"
rm -f "$LOG"

run() {  # run <tag> <extra args...>
    local tag="$1"; shift
    python "$HERE/regression_test.py" --tag "$tag" --out "$LOG" \
           --dump-logits "$OUT/$tag.npy" "$@" 2>/dev/null | tail -1
}

echo "########## 1) unchunked decoder (reference, weights saved) ##########"
run ref  --dec-chunk-codes 0 --save-weights "$W"

echo
echo "########## 2) chunked decoder, same weights ##########"
run chunked --dec-chunk-codes 1024 --load-weights "$W"

echo
echo "########## 3) compare logits ##########"
python - "$OUT/ref.npy" "$OUT/chunked.npy" <<'PY'
import sys
import numpy as np

a = np.load(sys.argv[1])
b = np.load(sys.argv[2])
if a.shape != b.shape:
    sys.exit("FAIL shapes differ: %s vs %s" % (a.shape, b.shape))
same = np.array_equal(a, b)
diff = float(np.abs(a - b).max())
print("   shape        : %s" % (a.shape,))
print("   max |a - b|  : %.3e" % diff)
print("   bit-identical: %s" % same)
sys.exit(0 if same else 1)
PY
rc=$?

echo
if [ $rc -eq 0 ]; then
    echo "PASS: chunking the code axis leaves the logits bit-identical."
else
    echo "FAIL: the chunked decoder differs from the unchunked one."
fi
echo "log: $LOG"
exit $rc
