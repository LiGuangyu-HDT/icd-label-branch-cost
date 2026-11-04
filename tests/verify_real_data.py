#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
verify_real_data.py -- first end-to-end check on real MIMIC-III data.

Verifies four things:
  1. the code table has 8,921 entries and 1,158 main codes, matching the official checkpoint;
  2. labels are no longer all zero, which is the one reliable signal that the LABELS format
     was converted correctly;
  3. one forward and backward pass runs and the loss is finite;
  4. peak memory at real scale.

Usage:
    export ICD_DATA_DIR=/path/to/mimic3_caml
    python verify_real_data.py --combiner lstm --rnn_dim 256 --num_layers 2 \
        --attention_head 4 --term_count 4            # full MSMN configuration
    python verify_real_data.py --varlen --bf16 ...   # varlen (flash) configuration
"""
import argparse, os, sys, time

import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("ICD_REPO", os.path.join(HERE, "..", "model", "main"))
sys.path.insert(0, REPO)

ap = argparse.ArgumentParser()
ap.add_argument("--version", default="mimic3")
ap.add_argument("--combiner", default="lstm")
ap.add_argument("--rnn_dim", type=int, default=256)
ap.add_argument("--num_layers", type=int, default=2)
ap.add_argument("--attention_head", type=int, default=4)
ap.add_argument("--term_count", type=int, default=4)
ap.add_argument("--attention_dim", type=int, default=512)
ap.add_argument("--truncate_length", type=int, default=4000)
ap.add_argument("--batch_size", type=int, default=8)
ap.add_argument("--num_transformer_layers", type=int, default=4)
ap.add_argument("--pos_embed", default="learn")
ap.add_argument("--attn_impl", default="naive")
ap.add_argument("--varlen", action="store_true")
ap.add_argument("--bf16", action="store_true")
ap.add_argument("--n_probe", type=int, default=4, help="how many samples to inspect")
a = ap.parse_args()

WE = os.environ.get("ICD_WORD_EMBEDDING", "")   # required: word2vec model path
dev = "cuda"

from train_parser import generate_parser                      # noqa: E402
from train_utils import generate_model                        # noqa: E402
from data_util import MimicFullDataset, my_collate_fn         # noqa: E402

O = "=" * 78
print(O); print("1) building the datasets (real data; takes a moment)"); print(O)

parser = generate_parser()
argv = ["--version", a.version, "--combiner", a.combiner, "--rnn_dim", str(a.rnn_dim),
        "--attention_dim", str(a.attention_dim), "--attention_head", str(a.attention_head),
        "--term_count", str(a.term_count), "--truncate_length", str(a.truncate_length),
        "--decoder", "MultiLabelMultiHeadLAATV2", "--word_embedding_path", WE,
        "--attn_impl", a.attn_impl, "--pos_embed", a.pos_embed]
if a.combiner == "lstm":
    argv += ["--num_layers", str(a.num_layers)]
else:
    argv += ["--num_transformer_layers", str(a.num_transformer_layers)]
if a.varlen:
    argv.append("--varlen")
if a.bf16:
    argv.append("--bf16")
args = parser.parse_args(argv)

t0 = time.time()
train_ds = MimicFullDataset(args.version, "train", WE, args.truncate_length,
                            args.label_truncate_length, args.term_count, args.sort_method)
print(f"  build time {time.time()-t0:.1f}s")
print(f"  len(train_ds)      = {len(train_ds)}")
print(f"  code_count         = {train_ds.code_count}     {'OK' if train_ds.code_count==8921 else 'FAIL expected 8921'}")
print(f"  main_code_count    = {train_ds.main_code_count}  {'OK' if train_ds.main_code_count==1158 else 'FAIL expected 1158'}")
print(f"  label sequences (c_input_word) = {train_ds.c_input_word.shape}")

print()
print(O); print("2) label check (the reliable signal for the LABELS format bug)"); print(O)
bad = 0
for i in range(min(a.n_probe, len(train_ds))):
    item = train_ds[i]
    lab = item[3] if len(item) > 3 else None
    s = int(sum(lab)) if lab is not None else -1
    if s <= 0:
        bad += 1
    print(f"  sample #{i:<5} positive labels = {s:>3}   {'OK' if s>0 else 'FAIL all zero'}")
print(f"  inspected {a.n_probe} samples, {bad} all-zero  {'PASS' if bad==0 else 'FAIL'}")

print()
print(O); print("3) forward and backward"); print(O)
model = generate_model(args, train_ds).to(dev)
model.c_input_word = train_ds.c_input_word.to(dev)
model.c_word_mask = train_ds.c_word_mask.to(dev)
dl = DataLoader(train_ds, batch_size=a.batch_size, collate_fn=my_collate_fn,
                shuffle=False, num_workers=2)
batch = tuple(x.to(dev) for x in next(iter(dl)))
print(f"  batch: input_word {tuple(batch[0].shape)}  mask {tuple(batch[1].shape)}")

import contextlib

amp = torch.autocast("cuda", dtype=torch.bfloat16) if a.bf16 else contextlib.nullcontext()
model.train()
torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
t0 = time.time()
with amp:
    out = model(batch, rdrop=args.rdrop_alpha > 0.0)
    loss = out["loss"] if isinstance(out, dict) else out
    loss.backward()
torch.cuda.synchronize()
dt = time.time() - t0
peak = torch.cuda.max_memory_allocated() / 1e9
print(f"  loss        = {float(loss):.4f}   {'finite' if torch.isfinite(loss) else 'NaN/Inf'}")
print(f"  c_loss      = {float(out['c_loss']):.4f}")
if "kl_loss" in out:
    print(f"  kl_loss     = {float(out['kl_loss']):.4f}")
print(f"  step time   = {dt*1000:.0f} ms (includes first-call overhead)")
print(f"  peak memory = {peak:.2f} GB / 24 GB")
grad_nan = sum(1 for p in model.parameters()
               if p.grad is not None and not torch.isfinite(p.grad).all())
print(f"  tensors with NaN/Inf gradients: {grad_nan} {'OK' if grad_nan==0 else 'FAIL'}")
print()
print(O)
print("result:", "real data runs end to end" if (bad == 0 and torch.isfinite(loss)) else "problems found")
print(O)
