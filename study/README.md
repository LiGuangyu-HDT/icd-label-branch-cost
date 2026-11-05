# Study drivers

These are the scripts that produced the runs reported in the paper. They expect a prepared
data directory, the word2vec model, and (for the eight-cell study) one or two GPUs.

All of them take their paths from the environment, so nothing here is tied to a particular
machine:

```bash
export ICD_DATA_DIR=/path/to/mimic3_caml
export ICD_WORD_EMBEDDING=/path/to/word2vec_sg0_100.model
export ICD_REPO=$PWD/model/main            # default: ../model/main
export OUT_DIR=/path/to/results            # default: ./out/<script>
```

`run_matrix_v2.sh` writes its run directories to `OUT_DIR` (default `study/out/runs_v2`),
tags each one with the cell name, and records the mapping in `$OUT_DIR/INDEX.tsv`. That tag
is what `analysis/` and `figures/` match on, so a run launched by hand needs `--tag <cell>`.

| Script | What it runs | Paper |
|---|---|---|
| `run_matrix_v2.sh` | the eight cells at full scale, one variable per step | Tables 3–11, Figs 2–5 |
| `run_precision_probe.sh` | P1 / P2: bf16 on one branch only | §6.4, Table 8 |
| `run_seeds.sh` | four configurations × three seeds in the 50-code setting | §6.5, Table 9 |
| `run_warmup_probe.sh` | rules out "the Transformer was undertrained" | §7.7 |
| `run_speed_retest.sh` | contention-free speed numbers | §5.8 |
| `run_pipeline2.sh` | the orchestrated chain, both GPUs | the reported runs |

## The eight cells

| Cell | Encoder | Precision | Decoder | Note |
|---|---|---|---|---|
| 1 | BiLSTM 2×256 | fp32 | naive, code-axis chunked | reference baseline |
| 2 | BiLSTM 2×256 | bf16 | naive, code-axis chunked | 1→2 precision |
| 3 | Transformer 4×512 | bf16 | naive | 2→3 architecture |
| 4 | Transformer 4×512 | bf16 | mem-efficient | 3→4 decoder only (micro-benchmark) |
| 5 | Transformer 4×512 | bf16 | varlen + flash | 4→5 all attention sites |
| 6 | Transformer 4×512 | fp32 | naive | 3→6 precision |
| 7 | Transformer 4×512 + BiLSTM label | bf16 | naive | the collapse |
| 8 | Transformer 4×512 + BiLSTM label | label fp32 | naive | the repair |

Every cell shares one recipe (Table 3 of the paper): effective batch 16, AdamW at 5e-4,
weight decay 0.01, no warm-up, 20 epochs, R-Drop α = 5, four synonym slots. `--epochs`
changes it for plumbing checks; the default is the paper's 20, not 1.

Cells 7 and 8 run from `model/hybrid` (the script selects the tree per cell); cells 1–6 from
`model/main`. Cell 7 puts the text branch on flash (`--attn_impl varlen`) and gives the label
branch its own BiLSTM (`--label_combiner lstm`), which forces that branch to the padded
representation — so `--label_repr` is deliberately not passed. Cell 8 adds `--label_fp32`,
which cannot be combined with `--label_repr varlen`.

## Selected on dev, compared at matched epochs

Every cell is selected on **dev P@8**, and all cross-configuration claims compare cells at
the **same epoch**. Comparing each cell at its own best epoch would let the longest-running
cell win by construction.

## Seeds

Seeds are deliberately not fixed: model initialisation uses unseeded `np.random` and the
DataLoader shuffles, so repeated runs are independent samples. Standard deviations are
therefore independent-sample s.d., and a difference between two cells uses
`sqrt(s1²/n1 + s2²/n2)` rather than a paired test.

## A note on the timing columns

The `h/epoch` and `ms/step` columns were recorded while two cells shared one node, so each
cell was competing for CPU. They are indicative rather than precise; the numbers quoted in
the paper for kernel and encoder comparisons come from `run_speed_retest.sh`, which runs one
task at a time.
