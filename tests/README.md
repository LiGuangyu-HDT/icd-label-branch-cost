# Tests

Run these before trusting any GPU number. They are dataset-free unless noted.

| Script | What it checks | Needs | Paper |
|---|---|---|---|
| `cpu_smoke.sh` | the whole CPU half: cell arguments, run-directory join, tables, seed aggregation, buckets, figures | numpy, matplotlib | §7 |
| `check_cell_args.py` | every cell's flags are defined by the model tree that cell runs from | stdlib | §6.1 |
| `make_synthetic_runs.py` | writes the fake results tree `cpu_smoke.sh` uses | numpy | — |
| `test_enc_repr.py` | `forward_chunked` and `forward_bucketed` against `forward` (**fails on regression**) | CUDA, word2vec | Table 2, V1–V2 |
| `test_varlen.py` | ragged (varlen) path against padded+mask, same weights | CUDA, flash-attn | Table 2, V3 |
| `verify_decoder_equivalence.sh` | chunked vs unchunked decoder on identical weights | CUDA | Table 2, V4 |
| `regression_test.py` | bit-identical output across a change; naive vs mem-efficient/flash decoder | CUDA, data | §4.5 |
| `test_hybrid_equiv.py` | hybrid combiner wiring, weight sharing, `state_dict` round-trip | CUDA | §6.4 |
| `verify_attention_backend.py` | which SDPA backend is actually selected per dtype | CUDA | §4.5 |
| `verify_real_data.py` | end-to-end on real data: code counts, labels, loss, peak memory | CUDA, data | §5.6 |
| `verify_b16.sh` | the "one cell per GPU, batch 16" parallel mode | CUDA, data | §5.8 |

## Start here: the CPU smoke test

```bash
bash tests/cpu_smoke.sh                 # ~20 s, no GPU, no MIMIC-III, no checkpoint
```

It builds a synthetic results tree, then runs `summarize_results.py`,
`aggregate_seeds.py`, `bucket_metrics.py` and `make_figures.py` over it, and checks that all
eight cells are found. This is the join between training output and paper tables — a name
mismatch there is silent (a cell reports "not started") and costs an afternoon to diagnose,
so it is the one part with no excuse to be broken. The numbers it draws are synthetic: they
check the plumbing, not the science. Never quote them.

## Expected results (GPU tests)

Pass/fail is by exit status: each script prints a summary line and exits non-zero when a
comparison leaves its tolerance (`TOLERANCE` in `test_enc_repr.py`).

| Test | Expected |
|---|---|
| V1 chunking | gradient difference exactly `0.00e+00` (bit-identical), rel-tolerance `1e-4` |
| V2 length bucketing | max output difference `6.4e-07`, relative `6.3e-07` in fp32 — the same order as the reference implementation's own `5.1e-07`; rel-tolerance `1e-4` in fp32, `1e-2` in bf16 |
| V3 ragged / varlen | decoder logits identical (rel `< 5e-2`) |
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
`../model/main` relative to this folder; use `../model/hybrid` for the hybrid cells).

## A note on seeds

`np.random` is deliberately unseeded in this repository — the UNK/MASK embeddings and
`coder_init` draw from it — so seeding `torch` alone does not reproduce initialisation.
`regression_test.py` therefore compares runs through **fixed weight files**
(`--save-weights` / `--load-weights`) rather than a fixed seed.
