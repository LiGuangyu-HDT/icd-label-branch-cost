# Micro-benchmarks

Each script is self-contained and measures one thing. Most take minutes, not hours.
Everything runs on the target GPU; nothing reads cross-machine calibration values.

| Script | Measures | Paper |
|---|---|---|
| `bench_branches.py` | text / label / decoder branches separately; checks the FLOP model | Table 1(a), §3.1 |
| `bench_bucketed.py` | padded vs length-bucketed vs ragged; equivalence, memory, time | Table 1(b), §4.2 |
| `bench_enc_ckpt.py` | checkpointing granularity: layer-level vs token chunks | Table 1(b), §3.4 |
| `bench_operators.py` | five key operators on the target GPU, in TFLOP/s | §3.1 |
| `bench_encoders.py` | BiLSTM vs Transformer encoder at L=4000 | §6.2 |
| `bench_backends.py` | SDPA's efficient vs flash backend at fixed dtype | §6.2 |
| `bench_masked.py` | how a padding mask changes backend selection and timing | §4.5 |
| `bench_fa_decoder.py` | naive vs variant-1c decoder, with an equivalence self-check | §4.5 |
| `bench_step_2x4090.py` | one composed training step, in minutes per epoch | §5.8 |
| `bench_cell1.py` | cell 1 end to end: naive (OOM), chunked, and 1c decoders | §5.8 |
| `bench_precision_drift.py` | what bf16 perturbs: features, logits, gradients, buckets | §7 |

## Table 1 in particular

- **Panel (a)**, per-term cost, comes from `bench_branches.py` and `bench_bucketed.py`.
- **Panel (b)**, what recomputation granularity saves, comes from `bench_enc_ckpt.py`.
- The `1.50x` faster / `1.90x` smaller kernel result is `bench_operators.py` together with
  the B=16 re-measurement in `bench_backends.py`.

## Running

```bash
export ICD_DATA_DIR=/path/to/mimic3_caml
export ICD_WORD_EMBEDDING=/path/to/word2vec_sg0_100.model
export ICD_REPO=$PWD/model/main          # default: ../model/main

python bench/bench_operators.py          # no data needed
python bench/bench_branches.py --small   # reduced batch first
python bench/bench_enc_ckpt.py --mode both --dtype bf16
python bench/bench_bucketed.py --mode both --dtype both
```

`--dry_data` / `--dry-data` substitutes synthetic inputs so the script itself can be smoke
tested without MIMIC.

## A caution on fp32 + varlen

`flash-attn` ships fp16/bf16 kernels only, so **`--dtype fp32` with the varlen path is not a
legal configuration**. Scripts that report it as failing are reporting a design constraint,
not a bug: what the fp32 rows measure is the dense (bucketed or padded) path.
