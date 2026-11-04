## 0. Provenance

The code in `model/main` and `model/hybrid` is **modified from the publicly released MSMN
implementation** that this paper builds on. That original distribution carries **no declared
licence**, so no licence is asserted over the portions derived from it; it is redistributed
here, in modified form, solely so that the results in this paper can be reproduced.

The modifications are the subject of the paper and are described in §4 and §5: exact
length-bucketing, ragged (varlen) encoding, token-level chunking of the description encoder,
code-axis chunking of the 8,921-way decoder, explicit attention-backend pinning, the hybrid
label combiner, and the instrumentation used for the measurements.

The MIT licence in `LICENSE` covers **our original contributions only**: the data-preparation
scripts, the equivalence test suite, the benchmarks, the study drivers and the analysis and
figure scripts. If you intend to redistribute the modified model code, contact the original
authors first.

---

# Code for *The label side is the bottleneck*

Reproduction code for the paper **"The label side is the bottleneck: memory cost, precision
sensitivity and a ranking reversal in ICD coding"**.

The paper is a *controlled diagnostic* study, not a leaderboard entry. This repository
contains exactly what is needed to reproduce its numbers: the two model variants, the data
preparation scripts, the equivalence test suite, the micro-benchmarks behind Table 1, the
drivers for the eight-cell study and the 50-code seed study, and the scripts that regenerate
every table and figure.

---

## 1. What is in here

```
.
├── model/
│   ├── main/               full-scale pipeline  — cells 1, 2, 3, 5, 6
│   └── hybrid/             hybrid label branch   — cells 7, 8
├── data/
│   ├── README.md           order of operations and what each list contains
│   ├── prepare_data.py     MIMIC-III extract -> the JSON/CSV the model reads (full setting)
│   ├── prepare_mimic3_50.py  the 200-slot MIMIC-III-50 setting
│   ├── official_codes.txt  the 8,921 ICD-9 codes of the full setting
│   └── splits/             the exact admission lists used (§2)
├── tests/                  numerical-equivalence and unit tests (Table 2, V1–V4), with README
├── bench/                  micro-benchmarks (Table 1 and the timings in §3–§7), with README
├── study/                  the controlled-study drivers, with README
├── analysis/               aggregation, long-tail bucketing, table assembly, with README
├── figures/                Fig. 1–5, with README
├── requirements.txt
├── LICENSE                 MIT (see §10 for what that does and does not cover)
├── .gitignore
└── README.md
```

`model/main` and `model/hybrid` differ only in `main.py`, `model/icd_model.py`,
`model/text_encoder.py`, `train_parser.py` and `train_utils.py`; everything else is shared.
Consecutive cells differ in **one factor only** — that is what makes the comparisons
attributable.

---

## 2. Data

MIMIC-III v1.4 is **not redistributable**. Obtain it from PhysioNet under the
[Credentialed Health Data License 1.5.0](https://physionet.org/content/mimiciii/1.4/) and
place the extract somewhere outside the repository and point `--src` at it
(conventionally `data/MIMIC-III/`, which `.gitignore` already excludes).

What this repository *does* ship is everything needed to pin the splits exactly:

| File | Rows | What it is |
|---|---|---|
| `splits/train_full_hadm_ids.csv` | 47,723 | caml-mimic official full-label train list |
| `splits/dev_full_hadm_ids.csv` | 1,631 | official dev list |
| `splits/test_full_hadm_ids.csv` | 3,372 | official test list |
| `splits/train_50_hadm_ids.csv` | 8,066 | official MIMIC-III-50 train list |
| `splits/dev_50_hadm_ids.csv` | 1,573 | official MIMIC-III-50 dev list |
| `splits/test_50_hadm_ids.csv` | 1,729 | official MIMIC-III-50 test list |
| `splits/used_train_hadm_ids.txt` | 47,600 | the subset actually used (records carrying at least one of the 8,921 codes) |
| `splits/NOT_IN_LOCAL_EXTRACT_hadm_ids.txt` | — | admissions absent from a given local extract |
| `splits/DROPPED_unmappable_hadm_ids.txt` | — | records that could not be mapped to a label |

These split files contain **bare hospital-admission identifiers only**: one integer per line,
no header, no subject identifiers, no dates, no diagnoses. They are shipped so that the
splits in the paper can be reproduced exactly. They are meaningless without the MIMIC-III
database itself, which you must obtain yourself under the credentialed licence; nothing here
can be used to reconstruct any patient record.

Prepare the full setting:

```bash
python data/prepare_data.py --help     # see the argument list
```

This writes `train_full.csv` / `dev_full.csv` / `test_full.csv` and the `*.json` files the
dataloader reads, and prints the resulting counts. The 8,921-code label table is written
from the official list, so `C = 8,921` matches the paper exactly.

### 2.1 Files that must come from *your* MIMIC-III extract

The repository deliberately ships **no MIMIC-derived content**. `model/*/constant.py`
resolves `DATA_DIR` from the `ICD_DATA_DIR` environment variable; that directory must
contain, produced by `prepare_data.py` from your own credentialed extract:

```
$ICD_DATA_DIR/
├── mimic3/                     the per-split json produced by prepare_data.py
├── D_ICD_DIAGNOSES.csv         hosp/d_icd_diagnoses.csv from MIMIC-III
├── D_ICD_PROCEDURES.csv        hosp/d_icd_procedures.csv from MIMIC-III
└── ICD9_descriptions           the ICD-9 hierarchy file
```

These are covered by the PhysioNet credentialed licence and are not redistributable.

The second setting:

```bash
python data/prepare_mimic3_50.py --help
```

---

## 3. Environment

```bash
pip install -r requirements.txt
```

`flash-attn` has no prebuilt wheel for every Python/CUDA combination and may take 20-40
minutes to compile. If you hit a download that stalls on a large package, fetching the wheel
with `curl -C -` and installing it locally is far more reliable than letting pip stream it;
this was the single biggest source of setup time when the environment was rebuilt.

Verified on **PyTorch 2.11.0+cu128 with flash-attn 2.8.3.post1**, Python 3.10.

Every result in the paper was produced on **2 × RTX 4090 (24 GB)**. The full-scale
configurations fit on a **single 24 GB device** after the zero-waste encoder of §4, which is
the point of the implementation.

The two environment variables the drivers expect:

```bash
export ICD_DATA_DIR=/path/to/data_caml          # output of prepare_data.py
export ICD_WORD_EMBEDDING=/path/to/word2vec_sg0_100.model
```

---

## 4. Numerical equivalence (Table 2)

Run these **before** trusting any GPU number. They are dataset-free.

```bash
python tests/test_enc_repr.py          # V1 / V2: chunking and length-bucketing
python tests/test_varlen.py            # V3: ragged (varlen) vs padded+mask
python tests/test_hybrid_equiv.py      # hybrid label combiner
python tests/regression_test.py        # end-to-end regression
bash   tests/verify_decoder_equivalence.sh   # V4: chunked code axis vs naive decoder
bash   tests/verify_b16.sh
```

Expected (paper, Table 2):

| Test | Result |
|---|---|
| V1 chunking | gradient difference exactly `0.00e+00` (bit-identical) |
| V2 length bucketing | max output difference `6.4e-07`, relative `6.3e-07` in fp32 — the same order as the reference implementation's own `5.1e-07` |
| V3 ragged / varlen | decoder logits identical |
| V4 decoder chunking | bit-identical |

---

## 5. Micro-benchmarks (Table 1, and the timings in §3–§4)

All of these run a single process with one batch layout, which is what makes the timings
comparable. They take minutes, not hours.

```bash
python bench/bench_branches.py         # per-term time and memory: text / label / decoder
python bench/bench_bucketed.py         # padded vs length-bucketed vs ragged
python bench/bench_enc_ckpt.py         # checkpointing granularity: layer-level vs token chunks
python bench/bench_operators.py        # kernel-level timings on the target GPU
python bench/bench_encoders.py         # BiLSTM vs Transformer encoder
python bench/bench_backends.py         # efficient-SDPA backends at fixed dtype
python bench/bench_fa_decoder.py       # flash-attention on the decoder
python bench/bench_masked.py           # effect of the padding mask on backend selection
python bench/bench_step_2x4090.py      # one full training step, min/epoch
python bench/bench_cell1.py            # cell 1 end to end at the original recipe
python bench/bench_precision_drift.py  # what bf16 actually perturbs (§7)
```

Table 1 in the paper is assembled from `bench_branches`, `bench_bucketed` and
`bench_enc_ckpt`; the `1.50× / 1.90×` kernel result is `bench_operators` together with the
B=16 re-measurement in `bench_backends`.

---

## 6. The controlled study

### 6.1 Eight cells at full scale (Tables 3–11, Fig. 2–5)

```bash
cd study
bash run_matrix_v2.sh --dry          # 5 steps per cell: real s/step and peak memory
bash run_matrix_v2.sh                # the full run
bash run_matrix_v2.sh --cells 3,6    # a subset
bash run_matrix_v2.sh --1gpu         # single device
```

The recipe is fixed across all cells (batch 16 effective, AdamW lr 5e-4, weight decay 0.01,
20 epochs, R-Drop α = 5, four synonym slots). Every cell inherits it unchanged; the paper's
Table 3 lists it in full.

### 6.2 Single-branch precision probes (§6.4, Table 8)

```bash
bash run_precision_probe.sh          # P1 = text bf16 / label fp32, P2 = the reverse
```

### 6.3 MIMIC-III-50, three seeds (§6.5, Table 9)

```bash
bash run_seeds.sh                    # four configurations × three seeds
```

Seeds are deliberately **not** fixed: model initialisation and shuffling are unseeded, so
repeat runs are independent samples rather than paired ones. Standard deviations are
therefore computed as independent-sample s.d., and differences as
`sqrt(s1²/n1 + s2²/n2)`.

### 6.4 Model selection rule

Every cell is selected on **dev P@8**, and all cross-configuration comparisons are made at
**matched epochs**, not at each cell's best epoch — otherwise a longer-running cell wins by
construction. Test thresholds come from dev, never from test.

---

## 7. Tables and figures

```bash
python analysis/summarize_results.py     # per-cell summary tables
python analysis/aggregate_seeds.py       # mean ± s.d. across seeds
python analysis/dump_predictions.py      # per-code predictions, needed for the long tail
python analysis/bucket_metrics.py        # five-way training-frequency buckets (Table 11)
python figures/make_figures.py           # Fig. 1–5 (reads the summaries; no GPU)
```

`figures/make_figures.py` writes `fig1.pdf … fig5.pdf`; Fig. 1 also carries the
`5.34×` / `94 KB per token` annotations, which are computed from the same summary files.

---

## 8. What is *not* reproduced here

- **Published systems are not re-run.** Table 10 places our cell 1 among the numbers
  reported by CAML, MultiResCNN, LAAT, RAC, LD-PLAM and PLM-ICD; the comparison is
  contextual, not a head-to-head re-implementation, and the paper says so.
- **Language-model coders are not evaluated.** Their published evaluations are on the small
  label setting, where the effects characterised here are largely absent.
- **MIMIC-IV / ICD-10 are not used.** Every result is on MIMIC-III v1.4 (ICD-9).

---

## 9. Hardware and cost

The full study consumed approximately 420 GPU-hours on two RTX 4090s. A single full-scale cell is
6–72 GPU-hours; a MIMIC-III-50 run is 0.4–3.3 hours, which is what makes three seeds
affordable there.

## 10. Licence

- **Our code** — `data/`, `tests/`, `bench/`, `study/`, `analysis/`, `figures/` and the
  modifications described in §0 — is released under the **MIT licence** (see `LICENSE`).
- **The model code** in `model/main` and `model/hybrid` is derived from the MSMN
  implementation, which carries **no declared licence**. It is included, unmodified in its
  original parts and modified only as described in the paper, **for reproduction purposes
  only**. No licence is asserted over those portions. Anyone wishing to redistribute them
  should contact the original authors.
- **MIMIC-III** is distributed by PhysioNet under the Credentialed Health Data License
  1.5.0. No MIMIC-derived file is included in this repository, and none may be
  redistributed with it.

If you use this code, please cite the paper.
