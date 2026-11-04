# Analysis

From run directories to the numbers and tables in the paper.

```bash
export ICD_RUNS=/path/to/runs            # default: ../runs
export ICD_DATA_DIR=/path/to/mimic3_caml
export ICD_WORD_EMBEDDING=/path/to/word2vec_sg0_100.model
```

| Script | Produces | Paper |
|---|---|---|
| `summarize_results.py` | `tables.md`, `tables.tex`, `epochs.csv`, per-cell summary | Tables 5–11 |
| `aggregate_seeds.py` | mean ± s.d. across seeds, and the difference with its standard error | Table 9, §6.3 |
| `dump_predictions.py` | `y_true/y_prob/y_pred/train_freq.npy` per cell, plus `meta.json` | needed by the long tail |
| `bucket_metrics.py` | macro/micro F1 and AUC per training-frequency bucket | Table 11 |

## Two conventions these scripts enforce

**Epoch selection.** Every cell is read at the epoch with the highest **dev P@8**, and the
test metrics of that epoch are reported. Thresholds come from dev, never from test.

**Independent samples.** Seeds are not fixed anywhere in this repository, so repeated runs
are independent samples rather than a paired design. `aggregate_seeds.py` therefore reports
differences with `sqrt(s1²/n1 + s2²/n2)` and refuses to compute a per-seed paired spread.

## Order

```bash
python analysis/summarize_results.py --runs "$ICD_RUNS" --out out/
python analysis/aggregate_seeds.py
python analysis/dump_predictions.py --all --runs "$ICD_RUNS" --out out/longtail   # needs GPU
python analysis/bucket_metrics.py --model "cell 1:out/longtail/cell1/y_true.npy,..." \
    --train-freq out/longtail/cell1/train_freq.npy --out out/latex_rows.tex
```

`dump_predictions.py` works out which repository the checkpoint came from, because the
hybrid cells pickle the whole `nn.Module` rather than a `state_dict`; loading one with the
wrong class computes the wrong forward pass without raising. The script prints the class and
file it used and fails loudly if they do not match.
