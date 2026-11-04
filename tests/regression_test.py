#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
regression_test.py -- regression and equivalence test for the ICD-MSMN modifications.

Two uses:

  1) Regression: the default path (naive decoder) must produce bit-identical output
     before and after a change.
         python regression_test.py --tag before
         python regression_test.py --tag after        # compare hashes with before

  2) Equivalence: the memory-efficient and flash decoders must be numerically equivalent
     to the naive one.
         python regression_test.py --attn-impl mem_efficient --tag me
         python regression_test.py --attn-impl flash         --tag fl

Output: one line of JSON holding a parameter hash and an output hash, so runs can be
diffed. Everything runs in eval mode under no_grad, so dropout cannot make it vary.
"""
import argparse, hashlib, json, os, random, sys
from contextlib import nullcontext

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("ICD_REPO", os.path.join(HERE, "..", "model", "main"))
sys.path.insert(0, REPO)

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="run")
ap.add_argument("--attn-impl", default=None, choices=["naive", "mem_efficient", "flash"])
ap.add_argument("--combiner", default="lstm", choices=["lstm", "transformer"])
ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--split", default="dev", choices=["train", "dev", "test"])
ap.add_argument("--out", default="")
ap.add_argument("--save-weights", default="", help="save the model weights here (for cross-implementation comparison)")
ap.add_argument("--load-weights", default="", help="load weights from this file, bypassing initialisation randomness")
ap.add_argument("--dump-logits", default="", help="dump logits to .npy for element-wise comparison")
ap.add_argument("--dec-chunk-codes", type=int, default=None,
                help="override dec_chunk_codes to compare chunked against unchunked decoding "
                     "on identical weights (Table 2, V4)")
a = ap.parse_args()


def seed_all(seed):
    """np.random is unseeded in this repository (the UNK/MASK embeddings and coder_init use
    it), so seeding torch alone is not enough."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

from train_parser import generate_parser                     # noqa: E402
from train_utils import generate_model                       # noqa: E402
from data_util import MimicFullDataset, my_collate_fn        # noqa: E402

WE = os.environ.get("ICD_WORD_EMBEDDING", "")   # required: word2vec model path

# ------------------------------------------------------------------ arguments
parser = generate_parser()
dests = {act.dest for act in parser._actions}
argv = ["--version", "mimic3-50", "--combiner", a.combiner,
        "--rnn_dim", "512", "--decoder", "MultiLabelMultiHeadLAATV2",
        "--attention_head", "8", "--attention_dim", "512",
        "--learning_rate", "5e-4", "--train_epoch", "1", "--batch_size", "16",
        "--xavier", "--main_code_loss_weight", "0.0", "--rdrop_alpha", "5.0",
        "--est_cls", "1", "--term_count", "8",
        "--word_embedding_path", WE]
if a.combiner == "lstm":
    argv += ["--num_layers", "1"]
elif "num_transformer_layers" in dests:
    argv += ["--num_transformer_layers", "4"]
if "attn_impl" in dests and a.attn_impl:
    argv += ["--attn_impl", a.attn_impl]
if a.dec_chunk_codes is not None and "dec_chunk_codes" in dests:
    argv += ["--dec_chunk_codes", str(a.dec_chunk_codes)]
args = parser.parse_args(argv)

dev = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------------------------ data and model
seed_all(a.seed)
# The training set must be built with exactly main.py's arguments
# (label_truncate_length / term_count / sort_method), otherwise the code table and the
# decoder's expected sizes disagree. dev/test take truncate_length only, as elsewhere.
train_ds = MimicFullDataset(args.version, "train", WE, args.truncate_length,
                            args.label_truncate_length, args.term_count, args.sort_method)
eval_ds = MimicFullDataset(args.version, a.split, WE, args.truncate_length)
batch = tuple(t.to(dev) for t in my_collate_fn([eval_ds[i] for i in range(len(eval_ds))]))

seed_all(a.seed)
model = generate_model(args, train_ds).to(dev)
if a.load_weights:
    sd = torch.load(a.load_weights, map_location=dev, weights_only=False)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("# loaded weights from %s (missing=%d unexpected=%d)"
          % (a.load_weights, len(missing), len(unexpected)), file=sys.stderr)
if a.save_weights:
    torch.save(model.state_dict(), a.save_weights)
    print("# saved weights to %s" % a.save_weights, file=sys.stderr)
model.c_input_word = train_ds.c_input_word.to(dev)
model.c_word_mask = train_ds.c_word_mask.to(dev)

mdtype = torch.float32 if a.dtype == "fp32" else torch.bfloat16
if a.dtype == "bf16":
    model = model.to(mdtype)

model.eval()
amp = torch.autocast("cuda", dtype=mdtype) if a.dtype == "bf16" else nullcontext()
with torch.no_grad(), amp:
    model.calculate_label_hidden()
    pred = model.predict(batch)
    logits = pred["yhat_raw"]

# ------------------------------------------------------------------ fingerprints
def h_tensor(t):
    t = t.detach().float().cpu().contiguous()
    return hashlib.sha256(t.numpy().tobytes()).hexdigest()[:16]

def h_params(m):
    h = hashlib.sha256()
    for n, p in sorted(m.named_parameters()):
        h.update(n.encode())
        h.update(p.detach().float().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]

res = {
    "tag": a.tag,
    "attn_impl": a.attn_impl or "(default)",
    "combiner": a.combiner,
    "dtype": a.dtype,
    "split": a.split,
    "n_params": sum(p.numel() for p in model.parameters()),
    "param_hash": h_params(model),
    "logits_shape": list(logits.shape),
    "logits_hash": h_tensor(logits),
    "logits_mean": round(float(logits.float().mean()), 6),
    "logits_std": round(float(logits.float().std()), 6),
    "logits_absmax": round(float(logits.float().abs().max()), 6),
}
if a.dump_logits:
    np.save(a.dump_logits, logits.detach().float().cpu().numpy())

line = json.dumps(res, ensure_ascii=False)
print(line)
if a.out:
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(line + "\n")
