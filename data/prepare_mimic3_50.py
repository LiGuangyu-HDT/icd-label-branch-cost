#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""prepare_mimic3_50.py -- build the MIMIC-III-50 setting from a prepared full extract.

CPU only; touches no data outside --src.

The 50-code setting carries 50 x 4 = 200 description slots instead of 35,684, so the
label branch is roughly 50x cheaper and a run costs hours rather than days. That is what
makes three seeds per configuration affordable, and the three seeds are what the error
bars in the paper come from.

Construction rule (matches the MSMN / LAAT / CAML convention)
------------------------------------------------------------
preprocess/icd_50.py hard-codes the official 50 codes, and the original construction
keeps every document whose labels intersect that set, then restricts the labels to it.

The result is checkable: the original MSMN paper reports 8,066 / 1,573 / 1,729 documents
for the 50-code setting, against 47,723 / 1,631 / 3,372 for the full one. This script
compares its own counts against those numbers on every run.

Input  : mimic3_{train,dev,test}.json produced by prepare_data.py
Output : mimic3_50/
           mimic3/mimic3-50_{train,dev,test}.json   documents (TEXT + LABELS)
           mimic3/{train,dev,test}_50.csv           code tables, same format as the full setting
         plus links to D_ICD_DIAGNOSES.csv, D_ICD_PROCEDURES.csv and ICD9_descriptions,
         which load_code_descriptions() reads from ICD_DATA_DIR.

Usage:
    python3 prepare_mimic3_50.py            # build and verify
    python3 prepare_mimic3_50.py --report   # report only, write nothing
"""
import argparse
import json
import os
import re
import sys
from collections import Counter

# Paths resolve relative to this file, so the working directory does not matter.
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "mimic3_caml", "mimic3")   # output of prepare_data.py
HADM_DIR = os.path.join(HERE, "splits")         # official {split}_50_hadm_ids.csv
OUT_ROOT = os.path.join(HERE, "mimic3_50")
OUT = os.path.join(OUT_ROOT, "mimic3")
ICD50_PY = os.path.abspath(os.path.join(HERE, "..", "model", "main", "preprocess", "icd_50.py"))
assert os.path.exists(ICD50_PY), "icd_50.py not found at %s" % ICD50_PY

#: Official document counts of the 50-code setting (MSMN paper, Table 1), used to verify
OFFICIAL_50 = {"train": 8066, "dev": 1573, "test": 1729}
OFFICIAL_FULL = {"train": 47723, "dev": 1631, "test": 3372}


def load_codes_50():
    src = open(ICD50_PY, encoding="utf-8").read()
    codes = re.findall(r"'([^']+)'", src)
    # Take the code list only (the first list literal in the file)
    start = src.index("codes_50")
    end = src.index("]", start)
    codes = re.findall(r"'([^']+)'", src[start:end])
    assert len(codes) == 50, "codes_50 should hold 50 codes, found %d" % len(codes)
    return codes


def mains():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true", help="report only; write nothing")
    ap.add_argument("--hadm-dir", default=HADM_DIR,
                    help="directory holding the official {split}_50_hadm_ids.csv lists. "
                         "The official lists must be used: filtering by label instead gives "
                         "train = 44,489 against the official 8,066")
    return ap.parse_args()


def main():
    a = mains()
    if not os.path.isdir(SRC):
        sys.exit("input directory not found: %s\n"
                 "Run prepare_data.py first, or point this script at its output." % SRC)
    codes50 = load_codes_50()
    print("=" * 74)
    print("Building the MIMIC-III-50 setting")
    print("=" * 74)
    print("official 50-code list (preprocess/icd_50.py, %d codes): %s ..." % (len(codes50), codes50[:5]))
    print("input directory: %s" % SRC)

    if not a.report:
        os.makedirs(OUT, exist_ok=True)

    # Used only for the frequency comparison below
    freq = Counter()

    for sp in ("train", "dev", "test"):
        src_path = os.path.join(SRC, "mimic3_%s.json" % sp)
        data = json.load(open(src_path, encoding="utf-8"))

        # The official admission lists must be used. Filtering by "labels intersect the
        # 50 codes" instead gives train = 44,489 against the official 8,066: the 50-code
        # setting has its own document split, it is not a label filter.
        hp = os.path.join(a.hadm_dir, "%s_50_hadm_ids.csv" % sp)
        official_ids = set()
        with open(hp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.lower().startswith("hadm"):
                    official_ids.add(line)
        print("\n--- %s: %d official entries (%s)" % (sp, len(official_ids), os.path.basename(hp)))

        kept, not_in_extract, no_50code = [], 0, 0
        seen = set()
        for rec in data:
            hid = str(int(rec["hadm_id"]))         # the official lists hold plain digit strings
            labs = rec["LABELS"]
            labs = labs.split(";") if isinstance(labs, str) else list(labs)
            labs = [c for c in labs if c]
            if sp == "train":
                freq.update(labs)
            if hid not in official_ids:
                continue
            seen.add(hid)
            inter = sorted(set(labs) & set(codes50))
            if not inter:
                no_50code += 1
                continue
            r = dict(rec)
            r["LABELS"] = ";".join(inter)
            kept.append(r)
        not_in_extract = len(official_ids - seen)

        print("    official %6d -> available locally %6d (missing %d, no 50-code label %d)"
              % (len(official_ids), len(kept), not_in_extract, no_50code))
        if not a.report:
            outp = os.path.join(OUT, "mimic3-50_%s.json" % sp)
            with open(outp, "w", encoding="utf-8") as f:
                json.dump(kept, f, ensure_ascii=False)
            print("    wrote %s (%.1f MB)" % (outp, os.path.getsize(outp) / 1e6))
        # Keep the official lists alongside the output
        dst = os.path.join(OUT, "%s_50_hadm_ids.csv" % sp)
        with open(dst, "w", encoding="utf-8") as f:
            for hid in sorted(official_ids, key=lambda x: int(x) if x.isdigit() else 0):
                f.write(hid + "\n")

    # Code-table CSV in the same format as the full setting (column 3 = LABELS); train
    # lists the 50 codes, dev and test carry the header only
    print("\n--- code-table CSV (load_full_codes reads column 3)")
    for sp in ("train", "dev", "test"):
        rows = codes50 if sp == "train" else []
        if not a.report:
            p = os.path.join(OUT, "%s_50.csv" % sp)
            with open(p, "w", encoding="utf-8") as f:
                f.write("HADM_ID,SUBJECT_ID,TEXT,LABELS\n")
                for c in rows:
                    f.write('0,0,,"%s"\n' % c)
            print("    %s : %d rows" % (os.path.basename(p), len(rows)))

    # Description files (load_code_descriptions reads them from ICD_DATA_DIR)
    print("\n--- description files (linked, not copied)")
    srcroot = os.path.join(HERE, "mimic3_caml")
    for name in ("D_ICD_DIAGNOSES.csv", "D_ICD_PROCEDURES.csv", "ICD9_descriptions"):
        s = os.path.join(srcroot, name)
        d = os.path.join(OUT_ROOT, name)
        if not os.path.exists(s):
            print("    [warn] source missing, skipped: %s" % s)
            continue
        if a.report:
            print("    %s -> %s" % (name, d))
            continue
        if os.path.islink(d) or os.path.exists(d):
            os.remove(d)
        os.symlink(s, d)
        print("    %s -> %s" % (name, d))

    # Diagnostics: which of the 50 codes never occur in our data
    print("\n" + "=" * 74)
    print("Diagnostics")
    print("=" * 74)
    missing = [c for c in codes50 if freq.get(c, 0) == 0]
    print("50 codes never seen in train: %d %s" % (len(missing), missing[:10]))
    top50 = [c for c, _ in freq.most_common(50)]
    only_official = [c for c in codes50 if c not in top50]
    print("official 50 codes outside our own top-50 by frequency: %d" % len(only_official))
    print("  e.g. %s" % only_official[:10])
    print("our top-10 codes by frequency: %s" % [c for c, _ in freq.most_common(10)])
    print("\nIf both counts above are small, the official 50-code list agrees with this data "
          "and the construction is sound.")
    print("output directory: %s" % OUT_ROOT)
    if not a.report:
        print("\nusage: export ICD_DATA_DIR=%s  then  --version mimic3-50" % OUT_ROOT)


if __name__ == "__main__":
    main()
