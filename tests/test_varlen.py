#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
test_varlen.py -- check that the varlen path is numerically equivalent to padded+mask.

Both paths must be compared under the same weights, so no second model is built:
  * encoder: one VarlenEncoderLayer, with a padded+SDPA+mask reference forward written by
    hand from that layer's own weights;
  * decoder: one decoder instance, first run as naive (padded), then with attn_impl
    switched to varlen (ragged).

Shapes:
  padded  h_pad  : (B, L, D) with word_mask (B, L)
  ragged  h_flat : (SL, D), obtained as h_pad[mask]

Usage:  python test_varlen.py
"""
import os
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("ICD_REPO", os.path.join(HERE, "..", "model", "main"))
sys.path.insert(0, REPO)

O = "=" * 78
dev = "cuda"
torch.manual_seed(0)
RESULTS = []


def report(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print("   %-44s %s  %s" % (name, "✅" if ok else "❌", detail))


from model.varlen import (lengths_from_mask, lengths_to_cu_seqlens, unpad,
                          segment_position_ids, segment_max, segment_mean,
                          VarlenEncoderLayer, HAS_FLASH_ATTN)

if not HAS_FLASH_ATTN:
    print("[warn] flash-attn not installed, skipping the varlen tests")
    sys.exit(0)

# ===========================================================================
print(O); print("0) helpers"); print(O)

B, L = 4, 16
mask = torch.ones(B, L, device=dev, dtype=torch.long)
mask[0, 10:] = 0
mask[2, 7:] = 0
lens = lengths_from_mask(mask)
cu = lengths_to_cu_seqlens(lens)
report("lengths_from_mask", lens.tolist() == [10, 16, 7, 16], str(lens.tolist()))
report("lengths_to_cu_seqlens", cu.tolist() == [0, 10, 26, 33, 49], str(cu.tolist()))

pos = segment_position_ids(lens, dev)
# Segment boundaries: 0-9 / 10-25 / 26-32 / 33-48
ok = (pos[:10].tolist() == list(range(10))
      and pos[10:26].tolist() == list(range(16))
      and pos[26:33].tolist() == list(range(7))
      and pos[33:49].tolist() == list(range(16)))
report("segment_position_ids restarts within each segment", ok)

x_pad = torch.randn(B, L, 8, device=dev)
x_flat = unpad(x_pad, mask)
report("unpad shape", tuple(x_flat.shape) == (49, 8), str(tuple(x_flat.shape)))

neg = torch.full_like(x_pad, -1e9)
ref_max = torch.where(mask.unsqueeze(-1).bool(), x_pad, neg).max(dim=1)[0]
report("segment_max == padded max",
       torch.allclose(ref_max, segment_max(x_flat, cu, B), atol=1e-6))
ref_mean = (x_pad * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
report("segment_mean == padded mean",
       torch.allclose(ref_mean, segment_mean(x_flat, cu, B), atol=1e-6))

# ===========================================================================
print(); print(O)
print("1) encoder layer: VarlenEncoderLayer vs a padded+SDPA+mask reference on the same weights")
print(O)

d_model, nhead, ff = 64, 8, 128
head_dim = d_model // nhead


def make_batch(Bx, Lx, d, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    m = torch.ones(Bx, Lx, device=dev, dtype=torch.long)
    lengths = torch.randint(Lx // 2, Lx + 1, (Bx,), generator=g)
    for i, n in enumerate(lengths.tolist()):
        m[i, n:] = 0
    x = torch.randn(Bx, Lx, d, generator=g).to(dev)
    return x, m


def ref_padded(layer, x, m, dt):
    """Padded + SDPA + additive mask using the layer's own weights (scale 1/sqrt(head_dim))."""
    Bx, Lx, _ = x.shape
    q = layer.q_proj(x).view(Bx, Lx, nhead, head_dim).transpose(1, 2)
    k = layer.k_proj(x).view(Bx, Lx, nhead, head_dim).transpose(1, 2)
    v = layer.v_proj(x).view(Bx, Lx, nhead, head_dim).transpose(1, 2)
    am = torch.zeros(Bx, 1, 1, Lx, device=dev, dtype=dt).masked_fill(
        (m == 0)[:, None, None, :], torch.tensor(-1e6, device=dev, dtype=dt))
    a = F.scaled_dot_product_attention(q, k, v, attn_mask=am, scale=1.0 / head_dim ** 0.5)
    return layer.out_proj(a.transpose(1, 2).reshape(Bx, Lx, d_model))


def ref_forward(layer, x, m, dt):
    y = layer.norm1(x + layer.dropout1(ref_padded(layer, x, m, dt)))
    return layer.norm2(y + layer.dropout2(layer._ff(y)))


# varlen works in bf16/fp16 only: flash-attn ships no fp32 kernel
for dt, dtn in ((torch.bfloat16, "bf16"),):
    layer = VarlenEncoderLayer(d_model, nhead, ff, dropout=0.0).to(dev, dt).eval()
    x, m = make_batch(4, 24, d_model, seed=1)
    x = x.to(dt)
    lengths = lengths_from_mask(m)
    cu = lengths_to_cu_seqlens(lengths)
    maxlen = int(lengths.max().item())

    # ---- forward ----
    with torch.no_grad():
        y_ref = unpad(ref_forward(layer, x, m, dt), m)
        y_var = layer(unpad(x, m), cu, maxlen)
    denom = y_ref.float().abs().max().item() + 1e-9
    rel = (y_ref.float() - y_var.float()).abs().max().item() / denom
    report("encoder %s forward" % dtn, rel < (1e-5 if dt == torch.float32 else 5e-2),
           "relative max|d|=%.2e" % rel)

    # ---- backward: both paths differentiate the sum of squares over real positions only ----
    xv = unpad(x, m).clone().requires_grad_(True)
    layer(xv, cu, maxlen).float().pow(2).sum().backward()
    g_var = xv.grad.clone()

    xp = x.clone().requires_grad_(True)
    yp = ref_forward(layer, xp, m, dt)
    (yp * m.unsqueeze(-1)).float().pow(2).sum().backward()
    g_ref = unpad(xp.grad, m)

    denom_g = g_ref.float().abs().max().item() + 1e-9
    relg = (g_ref.float() - g_var.float()).abs().max().item() / denom_g
    report("encoder %s backward" % dtn, relg < (1e-4 if dt == torch.float32 else 1e-1),
           "relative max|d|=%.2e" % relg)

# ===========================================================================
print(); print(O)
print("2) decoder: naive(padded) vs varlen(ragged) on one instance")
print(O)

from train_utils import create_decoder_config
from model.decoder import create_decoder


class _DS:                       # stand-in train_dataset, only to feed create_decoder_config
    code_count = 64
    ind2c = {i: "D%03d" % i for i in range(64)}
    ind2mc = {i: "D%02d" % i for i in range(16)}


class _Args:
    decoder = "MultiLabelMultiHeadLAATV2"
    attention_dim = 64
    code_embedding_path = None
    xavier = False
    est_cls = 0
    att_dropout = 0.0
    attention_head = 4
    rep_dropout = 0.0
    head_pooling = "max"
    act_fn_name = "tanh"
    attn_vchunk = 64
    attn_impl = "naive"


def build_decoder(impl):
    a = _Args(); a.attn_impl = impl
    cfg = create_decoder_config(a, _DS())
    return create_decoder(cfg).to(dev, torch.bfloat16).eval(), cfg


Bd, Ld, Ad, Md = 3, 20, 64, 4
head_dim_d = Ad // Md
for impl in ("naive", "mem_efficient", "varlen"):
    dec, cfg = build_decoder(impl)
    m2 = torch.ones(Bd, Ld, device=dev, dtype=torch.long)
    m2[0, 15:] = 0
    m2[2, 9:] = 0
    h_pad = torch.randn(Bd, Ld, Ad, device=dev, dtype=torch.bfloat16)
    lf = torch.randn(16 * Md, Ad, device=dev, dtype=torch.bfloat16)

    with torch.no_grad():
        # reference: same instance and weights, forced to naive
        cr = dict(cfg); cr['attn_impl'] = 'naive'
        dec.decoder_config = cr
        ref = dec.get_label_queried_features(h_pad, m2, lf)
        # subject: switch back to the implementation under test
        dec.decoder_config = cfg
        h_in = unpad(h_pad, m2) if impl == 'varlen' else h_pad
        got = dec.get_label_queried_features(h_in, m2, lf)

    denom = ref.float().abs().max().item() + 1e-9
    rel = (ref.float() - got.float()).abs().max().item() / denom
    report("decoder attn_impl=%-13s" % impl, rel < 5e-2,
           "relative max|d|=%.2e  shape %s" % (rel, tuple(got.shape)))

# ===========================================================================
print(); print(O)
n_ok = sum(RESULTS)
print("summary: %d/%d passed" % (n_ok, len(RESULTS)))
print(O)
sys.exit(0 if n_ok == len(RESULTS) else 1)
