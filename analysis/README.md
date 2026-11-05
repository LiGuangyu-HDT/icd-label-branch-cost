# Analysis

From run directories to the numbers and tables in the paper.

```bash
export ICD_RESULTS_ROOT=/path/to/results   # a dir holding runs_v2/, runs_seeds/, ...
export ICD_DATA_DIR=/path/to/mimic3_caml   # needed by dump_predictions.py only
export ICD_WORD_EMBEDDING=/path/to/word2vec_sg0_100.model
```

`--root` on any script overrides `ICD_RESULTS_ROOT`. Without either, the root is resolved
the way `figures/make_figures.py` resolves it (the paper's mirror layout, then the repository
itself). `study/run_matrix_v2.sh` writes to `<repo>/study/out/runs_v2` by default, so when
you analyse a fresh study run pass `--root study/out`.

| Script | Produces | Paper |
|---|---|---|
| `summarize_results.py` | `tables.md`, `tables.tex`, `epochs.csv`, per-cell summary (cells 1–8) | Tables 3–11 |
| `aggregate_seeds.py` | mean ± s.d. across seeds, and the difference with its standard error | Table 9, §6.3 |
| `dump_predictions.py` | `y_true/y_prob/y_pred/train_freq.npy` per cell, plus `meta.json` | needed by the long tail |
| `bucket_metrics.py` | macro/micro F1 and AUC per training-frequency bucket; `--json-out` feeds Fig. 5 | Table 11, Fig. 5 |

## Two conventions these scripts enforce

**Epoch selection.** Every cell is read at the epoch with the highest **dev P@8**, and the
test metrics of that epoch are reported. Thresholds come from dev, never from test.

**Independent samples.** Seeds are not fixed anywhere in this repository, so repeated runs
are independent samples rather than a paired design. `aggregate_seeds.py` therefore reports
differences with `sqrt(s1²/n1 + s2²/n2)` and refuses to compute a per-seed paired spread.

## Order

```bash
python analysis/summarize_results.py --runs "$ICD_RESULTS_ROOT/runs_v2" --out out/
python analysis/aggregate_seeds.py --root "$ICD_RESULTS_ROOT"
python analysis/dump_predictions.py --all --runs "$ICD_RESULTS_ROOT/runs_v2" --out out/longtail
python analysis/bucket_metrics.py --model "cell1:out/longtail/cell1/y_true.npy,..." \
    --train-freq out/longtail/cell1/train_freq.npy --out out/latex_rows.tex \
    --json-out "$ICD_RESULTS_ROOT/longtail/bucket_metrics.json"
python figures/make_figures.py --root "$ICD_RESULTS_ROOT" --out out/figures
```

`dump_predictions.py` works out which repository the checkpoint came from, because the
hybrid cells pickle the whole `nn.Module` rather than a `state_dict`; loading one with the
wrong class computes the wrong forward pass without raising. The script prints the class and
file it used and fails loudly if they do not match.

## Checking the join without any of that

These scripts need *run directories in the right shape*, not a model. To verify that they
find them — and that the names `study/` produces still match what they look for — run

```bash
bash tests/cpu_smoke.sh
```

which builds a synthetic tree, runs every script above, and fails if any of the eight cells
is not found. `--pod` additionally requires the `.cpod/rc` remote-copy wrapper; when that
helper is absent the script says so instead of reporting every run as missing.
