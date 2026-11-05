# Figures

```bash
python make_figures.py --root /path/to/results --out /path/to/figures
# writes fig1..fig5 as PDF and PNG
```

The results root is resolved in this order, so nothing here is tied to one machine:

1. `--root` on the command line,
2. `$ICD_RESULTS_ROOT`,
3. `<umbrella>/experiments/formal_results/experiments` (the layout the paper was made in),
4. the repository itself — what `study/run_matrix_v2.sh` writes with its default `OUT_DIR`.

| Figure | Content | Paper |
|---|---|---|
| `fig1.pdf` | where the cost goes, with the 5.34× and 94 KB/token annotations | Fig. 1, §3 |
| `fig2.pdf` | cost–accuracy landscape, marker area ∝ test AUC-macro | Fig. 2, §6.1 |
| `fig3.pdf` | precision × architecture, 2×2 at both scales | Fig. 3, §6.3–6.5 |
| `fig4.pdf` | label scale modulates precision sensitivity | Fig. 4, §6.5 |
| `fig5.pdf` | micro-F1 and AUC-macro by training-frequency bucket | Fig. 5, §6.8 |

Everything is generated from the run outputs; no experiment is re-run and no GPU is needed.

## What Fig. 5 reads

Fig. 5 takes its numbers from **`analysis/bucket_metrics.py --json-out`**, normally
`<root>/longtail/bucket_metrics.json`:

```bash
python3 analysis/bucket_metrics.py \
    --model 1_msmn_fp32_naive cell1/y_true.npy cell1/y_prob.npy cell1/y_pred.npy \
    --model 2_msmn_bf16_naive cell2/y_true.npy cell2/y_prob.npy cell2/y_pred.npy \
    --train-freq cell1/train_freq.npy \
    --out latex_rows.tex --json-out <root>/longtail/bucket_metrics.json
```

`--bucket-json` overrides the path. If the file is absent the script says so on stdout and
falls back to the values the *published* figure was drawn from — a re-run then cannot
silently keep a stale figure, but the fallback numbers are constants in the source, not
results.

## Exit status

`make_figures.py` exits non-zero when a figure could not be drawn (missing run directories)
or when `<root>/runs_v2` is absent. The figures that could be drawn are still written;
`--only 3,5` draws a subset.

## Two details worth knowing when reading the code

* **Cell 8's epoch time is a per-epoch median.** Its epochs 17–20 were contaminated by a
  co-scheduled job (775 ms against a steady 426 ms) and its selected epoch falls inside that
  window, so raw `epoch_seconds` cannot be used.
* **The Pareto frontier uses healthy cells only.** Cell 7 collapsed, so it is plotted but not
  treated as non-dominated.
