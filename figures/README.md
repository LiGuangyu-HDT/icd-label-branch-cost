# Figures

```bash
export ICD_RUNS=/path/to/runs     # default: ../runs
python make_figures.py            # writes fig1..fig5 as PDF and PNG
```

| Figure | Content | Paper |
|---|---|---|
| `fig1.pdf` | where the cost goes, with the 5.34× and 94 KB/token annotations | Fig. 1, §3 |
| `fig2.pdf` | cost–accuracy landscape, marker area ∝ test AUC-macro | Fig. 2, §6.1 |
| `fig3.pdf` | precision × architecture, 2×2 at both scales | Fig. 3, §6.3–6.5 |
| `fig4.pdf` | label scale modulates precision sensitivity | Fig. 4, §6.5 |
| `fig5.pdf` | micro-F1 and AUC-macro by training-frequency bucket | Fig. 5, §6.8 |

Everything is generated from the run outputs; no experiment is re-run and no GPU is needed.

Two details worth knowing when reading the code:

* **Cell 8's epoch time is a per-epoch median.** Its epochs 17–20 were contaminated by a
  co-scheduled job (775 ms against a steady 426 ms) and its selected epoch falls inside that
  window, so raw `epoch_seconds` cannot be used.
* **The Pareto frontier uses healthy cells only.** Cell 7 collapsed, so it is plotted but not
  treated as non-dominated.
