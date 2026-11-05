#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""check_cell_args.py -- verify that every cell's flags are accepted by the parser of the
model tree that cell runs from.

The failure this prevents: study/run_matrix_v2.sh grows a flag (`--label_fp32`,
`--label_combiner lstm`, ...) that the corresponding train_parser.py does not define, or a
cell starts pointing at the wrong tree. argparse rejects it only at launch time -- i.e. after
the data loader has started and, on a full-scale cell, hours into the run.

It reads the arguments from the script itself (`run_matrix_v2.sh --print-args`), so it
checks what will actually be executed, not a copy that can drift.

Usage:  python3 tests/check_cell_args.py
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
SCRIPT = os.path.join(REPO, "study", "run_matrix_v2.sh")


def cell_args():
    env = dict(os.environ, ICD_DATA_DIR="/dev/null", ICD_WORD_EMBEDDING="/dev/null")
    out = subprocess.run(["bash", SCRIPT, "--print-args"], capture_output=True,
                         text=True, env=env, cwd=REPO)
    if out.returncode != 0:
        sys.exit("run_matrix_v2.sh --print-args failed:\n%s%s" % (out.stdout, out.stderr))
    rows = []
    for line in out.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) == 4:
            rows.append(parts)
    if not rows:
        sys.exit("no cells were printed; did --print-args break?")
    return rows


def parser_for(repo):
    """The argparse parser of one model tree, imported in isolation."""
    repo = os.path.abspath(repo)
    for mod in ("train_parser",):
        sys.modules.pop(mod, None)
    sys.path.insert(0, repo)
    try:
        import train_parser                       # noqa: E402
        return train_parser.generate_parser()
    finally:
        sys.path.remove(repo)


def main():
    fails = 0
    print("=" * 78)
    print("check_cell_args: every cell's flags are defined by its own model tree")
    print("=" * 78)
    for cid, tag, repo, args in cell_args():
        parser = parser_for(repo)
        known, unknown = parser.parse_known_args(args.split())
        tree = os.path.basename(os.path.normpath(repo))
        bad = [u for u in unknown if u.startswith("--")]
        if bad:
            fails += 1
            print("  ❌ cell %s  %-30s (%s) does not define: %s"
                  % (cid, tag, tree, ", ".join(bad)))
        else:
            n = len([a for a in args.split() if a.startswith("--")])
            print("  ✅ cell %s  %-30s (%s)  %d flags accepted" % (cid, tag, tree, n))
            if tree == "hybrid" and getattr(known, "label_combiner", "same") == "same":
                print("     [warn] hybrid tree but --label_combiner is not set: the hybrid "
                      "branch is not actually used")

    print()
    if fails:
        print("%d cell(s) would fail at launch" % fails)
        return 1
    print("all cells parse")
    return 0


if __name__ == "__main__":
    sys.exit(main())
