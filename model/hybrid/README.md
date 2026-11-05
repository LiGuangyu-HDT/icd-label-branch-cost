# model/hybrid

Derived from the MSMN implementation (ACL 2022, *Code Synonyms Do Matter*) as described in
the repository's top-level README, section 0 (Provenance). The parts that differ from it are
the subject of the paper in this repository.

**Do not follow the environment instructions this file used to carry.** The upstream README
said Python 3.7 / PyTorch 1.7.0 and "at least 32 GB of GPU"; this release runs on
**Python 3.10 / PyTorch 2.11.0+cu128** and, with the zero-waste encoder of paper §4, the
full-scale configurations fit in a **single 24 GB device**.

Use:

    pip install -r ../../requirements.txt

Data preparation is in `data/` at the repository root, not in caml-mimic: see
`data/README.md`. MIMIC-III is not redistributed here, and no sample data ships with this
directory -- the upstream `sample_data/` was removed because it contained MIMIC-derived
files.

Which cells this variant runs, and the exact commands, are in the top-level `README.md`
(§6) and `study/README.md`.
