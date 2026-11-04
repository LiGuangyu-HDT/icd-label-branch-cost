# Tests

Run these before trusting any GPU number. They are dataset-free unless noted.

| Script | What it checks | Paper |
|---|---|---|
| `test_enc_repr.py` | `forward_chunked` and `forward_bucketed` against `forward` | Table 2, V1–V2 |
| `test_varlen.py` | ragged (varlen) path against padded+mask, same weights | Table 2, V3 |
| `verify_decoder_equivalence.sh` | chunked vs unchunked decoder on identical weights | Table 2, V4 |
| `regression_test.py` | bit-identical output across a change; naive vs mem-efficient/flash decoder | §4.5 |
| `test_hybrid_equiv.py` | hybrid combiner wiring, weight sharing, `state_dict` round-trip | §6.4 |
| `verify_attention_backend.py` | which SDPA backend is actually selected per dtype | §4.5 |
| `verify_real_data.py` | end-to-end on real data: code counts, labels, loss, peak memory | §5.6 |
| `verify_b16.sh` | the "one cell per GPU, batch 16" parallel mode | §5.8 |

## Expected results

| Test | Expected |
|---|---|
| V1 chunking | gradient difference exactly `0.00e+00` (bit-identical) |
| V2 length bucketing | max output difference `6.4e-07`, relative `6.3e-07` in fp32 — the same order as the reference implementation's own `5.1e-07` |
| V3 ragged / varlen | decoder logits identical |
| V4 decoder chunking | bit-identical logits |

## Running

Most scripts need only PyTorch:

```bash
python tests/test_enc_repr.py
python tests/test_varlen.py
python tests/test_hybrid_equiv.py
```

The ones that build a model need the data and embeddings:

```bash
export ICD_DATA_DIR=/path/to/mimic3_caml
export ICD_WORD_EMBEDDING=/path/to/word2vec_sg0_100.model
python tests/verify_real_data.py --combiner lstm --rnn_dim 256 --num_layers 2 \
    --attention_head 4 --term_count 4
bash   tests/verify_decoder_equivalence.sh
bash   tests/verify_b16.sh 3 6 100
```

`ICD_REPO` overrides where the scripts look for the model package (default
`../model/main` relative to this folder).

## A note on seeds

`np.random` is deliberately unseeded in this repository — the UNK/MASK embeddings and
`coder_init` draw from it — so seeding `torch` alone does not reproduce initialisation.
`regression_test.py` therefore compares runs through **fixed weight files**
(`--save-weights` / `--load-weights`) rather than a fixed seed.
