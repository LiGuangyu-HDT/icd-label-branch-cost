#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
prepare_data.py -- turn a raw MIMIC-III json export into the layout ICD-MSMN expects.

Reads only; nothing in the source directory is modified. Everything is written to --out
(default data_prepared/mimic3_caml).

Three jobs:

1. LABELS format. The raw json stores labels as a JSON list (``["820.21", "997.3"]``)
   while data_util.py splits a semicolon string. Left as a list, ``str(list).split(';')``
   yields a single token and every label silently becomes 0.

2. Split assignment. Records are assigned to the official caml-mimic admission lists
   (47,723 / 1,631 / 3,372) rather than to whatever split the export happened to carry.

3. Code table. train_full.csv / dev_full.csv / test_full.csv exist only to give
   load_full_codes() its label table, so the official 8,921-code list is written into
   them. Inferring codes from the json recovers 8,919 and differs from the official list
   in both directions (see --report), so the official table keeps C comparable.

Usage:
    python3 prepare_data.py                 # write the prepared data
    python3 prepare_data.py --report        # report differences only, write nothing
"""
import argparse
import os
import shutil
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="data", help="directory holding the raw json (read-only)")
ap.add_argument("--caml", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "splits"),
                help="official caml-mimic hadm_id lists (default: ./splits)")
ap.add_argument("--vocab", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                "official_codes.txt"),
                help="official 8,921-code table (default: ./official_codes.txt)")
ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             "mimic3_caml"),
                help="output directory (default: ./mimic3_caml)")
ap.add_argument("--icd-dir", default=None,
                help="directory holding D_ICD_DIAGNOSES.csv, D_ICD_PROCEDURES.csv and "
                     "ICD9_descriptions, which are copied next to the output. These come "
                     "from hosp/ of your own MIMIC-III extract and are not bundled here; "
                     "omit this flag to skip the copy")
ap.add_argument("--report", action="store_true", help="report only; write nothing")
ap.add_argument("--keep-unmapped", action="store_true",
                help="keep records whose labels all fall outside the official code table "
                     "(dropped by default; 8 records are affected, and dropping them keeps "
                     "the supervision clean and C at 8,921)")
args = ap.parse_args()

SPLITS = ("train", "dev", "test")


# --------------------------------------------------------------------------- #
def load_official_split(caml_dir):
    """Map an official hadm_id to its split."""
    m = {}
    for sp in SPLITS:
        p = os.path.join(caml_dir, f"{sp}_full_hadm_ids.csv")
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line.isdigit():
                    m[line] = sp
    return m


def iter_records(path):
    """Yield each record's raw lines, holding one record in memory at a time.

    The file is a pretty-printed JSON array; records start at ``    {`` and end at
    ``    }`` or ``    },``.
    """
    buf, inside = [], False
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not inside:
                if line.rstrip("\n") == "    {":
                    inside, buf = True, [line]
            else:
                buf.append(line)
                if line.rstrip("\n") in ("    },", "    }"):
                    yield buf
                    inside = False


def record_hadm(buf):
    for ln in buf:
        s = ln.strip()
        if s.startswith('"hadm_id"'):
            return s.split('"')[3]
    return None


def rewrite_labels(buf):
    """Convert LABELS from a list to a semicolon string and return the new lines.

    The trailing comma on the last line (``    },`` or ``    }``) is removed: separators
    are emitted by the writer instead, otherwise the output contains ``},,``.
    """
    out, i, n = [], 0, len(buf)
    while i < n:
        line = buf[i]
        if line.strip() == '"LABELS": [':
            codes, j = [], i + 1
            while j < n and buf[j].strip() != ']':
                codes.append(buf[j].strip().rstrip(',').strip('"'))
                j += 1
            # A trailing comma is needed only if another field follows LABELS.
            # The record-closing `    }` after `]` is not a field.
            has_more = any(buf[m].startswith('        "') for m in range(j + 1, n))
            out.append('        "LABELS": "%s"%s\n' % (';'.join(codes), ',' if has_more else ''))
            i = j + 1                      # skip `        ]`
        else:
            out.append(line)
            i += 1
    if out:
        out[-1] = out[-1].rstrip("\n").rstrip(",") + "\n"
    return out


def vocab_from_json(paths):
    """Recover the code table from the json LABELS (for the --report comparison)."""
    v = set()
    for p in paths:
        for buf in iter_records(p):
            hit = False
            for ln in buf:
                if ln.strip() == '"LABELS": [':
                    hit = True
                    continue
                if hit:
                    s = ln.strip()
                    if s == ']':
                        hit = False
                    elif s:
                        v.add(s.rstrip(',').strip('"'))
    return v


# --------------------------------------------------------------------------- #
def main():
    if not os.path.isdir(args.src):
        sys.exit(f"source directory not found: {args.src}")
    official = load_official_split(args.caml)
    with open(args.vocab) as f:
        official_codes = sorted(l.strip() for l in f if l.strip())
    print(f"official split entries : {len(official)}")
    print(f"official code table    : {len(official_codes)}")

    srcs = [os.path.join(args.src, f"mimic3_{s}.json") for s in SPLITS]
    for p in srcs:
        if not os.path.exists(p):
            sys.exit(f"not found: {p}")

    data_codes = vocab_from_json(srcs)
    only_off = sorted(set(official_codes) - data_codes)
    only_dat = sorted(data_codes - set(official_codes))
    print(f"codes inferred from json: {len(data_codes)}")
    print(f"  only in official ({len(only_off)}): {only_off}")
    print(f"  only in data     ({len(only_dat)}): {only_dat}")

    if args.report:
        return

    os.makedirs(args.out, exist_ok=True)
    outs = {sp: open(os.path.join(args.out, f"mimic3_{sp}.json"), "w", encoding="utf-8")
            for sp in SPLITS}
    for fh in outs.values():
        fh.write("[\n")

    vocab_set = set(official_codes)
    stat = {sp: {"n": 0, "first": True} for sp in SPLITS}
    unassigned = dropped = 0
    for si, p in enumerate(srcs):
        for buf in iter_records(p):
            hadm = record_hadm(buf)
            sp = official.get(hadm)
            if sp is None:
                unassigned += 1
                continue
            if not args.keep_unmapped:
                labs = []
                inl = False
                for ln in buf:
                    t = ln.strip()
                    if t == '"LABELS": [':
                        inl = True; continue
                    if inl:
                        if t == ']': break
                        labs.append(t.rstrip(',').strip('"'))
                if labs and not any(c in vocab_set for c in labs):
                    dropped += 1
                    continue
            st = stat[sp]
            if not st["first"]:
                outs[sp].write(",\n")
            st["first"] = False
            outs[sp].writelines(rewrite_labels(buf))
            st["n"] += 1

    for sp in SPLITS:
        outs[sp].write("\n]\n")
        outs[sp].close()

    print("\n=== output ===")
    for sp in SPLITS:
        print(f"  mimic3_{sp}.json : {stat[sp]['n']:>6}")
    print(f"  unassigned             : {unassigned}")
    print(f"  dropped (no mappable label): {dropped}")

    # --- CSV: code-table carrier -------------------------------------------- #
    # Training reads this CSV only through load_full_codes(), so writing the official
    # table here is what fixes C = 8,921.
    per = {"train": official_codes, "dev": [], "test": []}
    for sp in SPLITS:
        path = os.path.join(args.out, f"{sp}_full.csv")
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("HADM_ID,SUBJECT_ID,TEXT,LABELS\n")
            for c in per[sp]:
                f.write(f'0,0,,"{c}"\n')
        print(f"  {sp}_full.csv    : {len(per[sp])} rows")

    # --- accompanying files -------------------------------------------------- #
    if args.icd_dir:
        for name in ("D_ICD_DIAGNOSES.csv", "D_ICD_PROCEDURES.csv", "ICD9_descriptions"):
            src = os.path.join(args.icd_dir, name)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(args.out, name))
            else:
                print(f"  [warn] not found, skipped: {name}")
    else:
        print("  [note] --icd-dir not given: the ICD dictionaries were not copied. "
              "The model needs them in ICD_DATA_DIR.")
    print(f"\noutput directory: {args.out}")
    print("the source directory was not modified.")


if __name__ == "__main__":
    main()
