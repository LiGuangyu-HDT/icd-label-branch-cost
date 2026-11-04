# Data preparation

Two scripts and the admission lists they use. Nothing here redistributes MIMIC-III.

## Order of operations

```bash
# 1. full setting: raw json export  ->  mimic3_caml/
python data/prepare_data.py \
    --src /path/to/raw/json \
    --icd-dir /path/to/mimiciii/hosp \
    --vocab data/official_codes.txt \
    --out data/mimic3_caml

# 2. the 200-slot setting: mimic3_caml/  ->  mimic3_50/
python data/prepare_mimic3_50.py
```

Both take `--report` to print their checks without writing anything.

## Files

| File | What it is |
|---|---|
| `prepare_data.py` | raw json → the layout `ICD-MSMN` reads (LABELS format, official splits, code table) |
| `prepare_mimic3_50.py` | builds the MIMIC-III-50 setting from the full one |
| `official_codes.txt` | the 8,921 ICD-9 codes of the full setting, one per line |
| `splits/train_full_hadm_ids.csv` | official caml-mimic train list — 47,723 admissions |
| `splits/dev_full_hadm_ids.csv` | official dev list — 1,631 |
| `splits/test_full_hadm_ids.csv` | official test list — 3,372 |
| `splits/train_50_hadm_ids.csv` | official MIMIC-III-50 train list — 8,066 |
| `splits/dev_50_hadm_ids.csv` | official 50-code dev list — 1,573 |
| `splits/test_50_hadm_ids.csv` | official 50-code test list — 1,729 |
| `splits/used_train_hadm_ids.txt` | the 47,600 admissions actually used (those carrying at least one of the 8,921 codes) |
| `splits/used_dev_hadm_ids.txt` | 1,626 used of the official 1,631 |
| `splits/used_test_hadm_ids.txt` | 3,360 used of the official 3,372 |
| `splits/NOT_IN_LOCAL_EXTRACT_hadm_ids.txt` | 132 official admissions absent from a given local extract |
| `splits/DROPPED_unmappable_hadm_ids.txt` | 8 records whose labels all fall outside the official code table |

The counts in `used_*` are what the paper reports: 47,600 training admissions gives
47,600 / 16 = 2,975 optimisation steps per epoch.

## What must come from your own extract

`prepare_data.py --icd-dir` copies three files that are covered by the PhysioNet
credentialed licence and are therefore **not** in this repository:

```
D_ICD_DIAGNOSES.csv     from hosp/ of MIMIC-III
D_ICD_PROCEDURES.csv    from hosp/ of MIMIC-III
ICD9_descriptions       the ICD-9 hierarchy file
```

They end up next to the prepared output and must be present in `ICD_DATA_DIR` for the
model to load code descriptions.

## Code table

`official_codes.txt` holds the 8,921 code identifiers only — no patient data. It is
written into `train_full.csv` / `dev_full.csv` / `test_full.csv` because training reads
those CSVs solely through `load_full_codes()`, so this is what pins `C = 8,921`.
Inferring the codes from the json instead recovers 8,919 and differs from the official
list in both directions; `--report` prints that comparison.
