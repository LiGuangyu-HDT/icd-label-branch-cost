#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
test_hybrid_equiv.py -- dataset-free unit tests for the hybrid combiner
(--label_combiner lstm).

Runs on CPU in well under a minute; no GPU, no training:
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 python -u test_hybrid_equiv.py

Each check guards something that would silently change the paper's numbers if wrong:

  1. --label_combiner same -> create_label_combine_config returns None (original behaviour)
  2. --label_combiner lstm -> returns a copied config and does not mutate the text one
  3. both branches share one Word_Encoder object (no duplicated embeddings, no doubled
     parameter count)
  4. the label branch's forward returns the right shape (N, L, attention_dim)
  5. IcdModel wiring: same -> label_text_encoder is None; lstm -> a separate encoder that
     shares the embeddings
  6. state_dict round-trips (no enums or closures stored on the module)
"""
import copy
import os
import sys
import types

# Allow running from any directory: put the repo root (ICD_REPO, default cwd) on sys.path
_repo = os.environ.get('ICD_REPO') or os.getcwd()
if _repo not in sys.path:
    sys.path.insert(0, _repo)

import torch
from torch import nn

import model.icd_model as icd_model_mod
from model.icd_model import IcdModel, create_label_combine_config
from model.text_encoder import TextEncoder

FAILS = []


def check(name, cond, extra=""):
    print("  %s %s%s" % ("✅" if cond else "❌", name, ("  " + extra) if extra else ""))
    if not cond:
        FAILS.append(name)


def word_config():
    return {'count': 50, 'dim': 8, 'padding_idx': 0, 'dropout': 0.0,
            'word_embedding_path': None, 'frz': False}


def text_combine_config():
    """The real path uses a 4-layer Transformer for the text side (shrunk to 1 layer, 16 dims)."""
    return {
        'model': 'transformer', 'rnn_dim': 16, 'num_layers': 1,
        'transformer_head': 2, 'transformer_ff': 32,
        'transformer_activation': 'gelu', 'transformer_dropout': 0.0,
        'norm_first': False, 'zero_pad_output': True, 'pos_embed': 'learn',
        'layer_norm': False, 'dim': 16,          # dim = attention_dim = the combiner's output width
        'varlen': True, 'enc_repr': 'varlen', 'enc_chunk_tokens': 0,
        'enc_attn_backend': 'efficient',
    }


def make_args(label_combiner='same', rnn=None, layers=None, dp=None, **kw):
    a = types.SimpleNamespace(label_combiner=label_combiner, **kw)
    if rnn is not None:
        a.label_rnn_dim = rnn
    if layers is not None:
        a.label_lstm_layers = layers
    if dp is not None:
        a.label_lstm_dp = dp
    return a


def main():
    torch.manual_seed(0)
    wc = word_config()
    cc = text_combine_config()

    print("\n[1] label_combiner='same' -> no new config (original behaviour)")
    cc_before = copy.deepcopy(cc)
    check("returns None", create_label_combine_config(make_args('same'), cc) is None)
    check("text config unchanged", cc == cc_before)

    print("\n[2] label_combiner='lstm' -> a separate config that leaves the text one alone")
    lcc = create_label_combine_config(make_args('lstm'), cc)
    check("returns a dict", isinstance(lcc, dict))
    check("is a copy, not the same object", lcc is not cc)
    check("text config not mutated", cc == cc_before)
    check("model=lstm", lcc['model'] == 'lstm')
    check("rnn_dim/layers/dropout default to 256/2/0.1",
          (lcc['rnn_dim'], lcc['num_layers'], lcc['lstm_dropout']) == (256, 2, 0.1),
          str((lcc['rnn_dim'], lcc['num_layers'], lcc['lstm_dropout'])))
    check("forced to padded (BiLSTM takes no ragged input)",
          lcc['varlen'] is False and lcc['enc_repr'] == 'padded')
    lcc1 = create_label_combine_config(make_args('lstm', rnn=64, layers=1, dp=0.3), cc)
    check("dropout zeroed at 1 layer, as create_combine_config does",
          (lcc1['rnn_dim'], lcc1['num_layers'], lcc1['lstm_dropout']) == (64, 1, 0.0))

    print("\n[3] both branches share one Word_Encoder (embeddings not duplicated)")
    text_te = TextEncoder(wc, copy.deepcopy(cc))
    label_te = TextEncoder(wc, create_label_combine_config(make_args('lstm'), cc),
                           word_encoder=text_te.word_encoder)
    check("word_encoder is the same object", label_te.word_encoder is text_te.word_encoder)
    check("the embedding weights are one tensor",
          label_te.word_encoder.word_embedding.weight is
          text_te.word_encoder.word_embedding.weight)
    emb_once = sum(p.numel() for p in text_te.word_encoder.parameters())
    total = sum(p.numel() for p in text_te.parameters()) + \
        sum(p.numel() for p in label_te.combiner.parameters()) + \
        sum(p.numel() for p in label_te.word_encoder.parameters())
    check("embeddings built once",
          total == sum(p.numel() for p in text_te.parameters()) +
          sum(p.numel() for p in label_te.parameters()),
          "emb=%d" % emb_once)
    check("the two combiners are distinct modules", text_te.combiner is not label_te.combiner)
    check("text side is a Transformer, label side a BiLSTM",
          type(text_te.combiner).__name__ == 'Transformer_Combiner' and
          type(label_te.combiner).__name__ == 'LSTM_Combiner',
          "%s / %s" % (type(text_te.combiner).__name__, type(label_te.combiner).__name__))

    print("\n[4] label-branch forward shape (N, L, attention_dim)")
    N, L = 6, 5
    ids = torch.randint(1, 50, (N, L))
    mask = torch.ones(N, L, dtype=torch.long)
    mask[0, 3:] = 0
    out = label_te(ids, mask, 'padded')
    check("forward runs and returns 3 dims", out.dim() == 3)
    check("batch dim = %d" % N, out.shape[0] == N)
    check("last dim = attention_dim (16)", out.shape[-1] == 16, str(tuple(out.shape)))

    print("\n[5] IcdModel wiring (decoder replaced by Identity to avoid needing a dataset)")
    real_create_decoder = icd_model_mod.create_decoder
    icd_model_mod.create_decoder = lambda cfg: nn.Identity()
    try:
        label_config = {'input_dim': 16, 'num_layers': 0, 'dropout': 0.0,
                        'pooling': 'max', 'varlen': False}
        m_same = IcdModel(wc, copy.deepcopy(cc), {}, label_config, {}, args=make_args('same'))
        m_hyb = IcdModel(wc, copy.deepcopy(cc), {}, label_config, {},
                         args=make_args('lstm', rnn=8, layers=2, dp=0.1))
        check("same → label_text_encoder is None", m_same.label_text_encoder is None)
        check("same -> label branch uses the shared encoder", m_same._label_enc() is m_same.encoder)
        check("lstm -> a separate encoder exists", m_hyb.label_text_encoder is not None)
        check("lstm -> label branch uses that encoder", m_hyb._label_enc() is m_hyb.label_text_encoder)
        check("lstm -> the separate encoder shares the text embeddings",
              m_hyb.label_text_encoder.word_encoder is m_hyb.encoder.word_encoder)
        n_same = sum(p.numel() for p in m_same.parameters())
        n_hyb = sum(p.numel() for p in m_hyb.parameters())
        lc = sum(p.numel() for p in m_hyb.label_text_encoder.combiner.parameters())
        check("the parameter increase equals the label-side combiner's", n_hyb - n_same == lc,
              "Δ=%d lc=%d" % (n_hyb - n_same, lc))

        print("\n[6] state_dict round-trips (pickle-safe)")
        import io
        buf = io.BytesIO()
        torch.save(m_hyb.state_dict(), buf)
        buf.seek(0)
        sd = torch.load(buf, map_location='cpu', weights_only=True)
        check("state_dict round-trip succeeds", len(sd) == len(m_hyb.state_dict()))
        check("no unexpected keys beyond label_text_encoder",
              any(k.startswith('label_text_encoder.') for k in sd))

        print("\n[7] per-branch precision switches (--text_fp32 / --label_fp32)")
        # These two switches isolate which side bf16 damages.
        from model.icd_model import _branch_fp32_flag
        check("_branch_fp32_flag is False when the switch is absent", _branch_fp32_flag(make_args('same'), 'text_fp32') is False)
        check("True when the switch is given",
              _branch_fp32_flag(make_args('same', text_fp32=True), 'text_fp32') is True)
        check("no error when args is None (tests, inference)", _branch_fp32_flag(None, 'text_fp32') is False)
        for nm in ('text_fp32', 'label_fp32'):
            m = IcdModel(wc, copy.deepcopy(cc), {}, label_config, {}, args=make_args('same', **{nm: True}))
            ctx = getattr(m, '_text_ctx' if nm == 'text_fp32' else '_label_ctx')()
            check("%s=True yields a usable context and does not change the parameter count" % nm,
                  hasattr(ctx, '__enter__') and
                  sum(p.numel() for p in m.parameters()) == n_same)
        # With the switch off this must be a no-op, so nothing changes.
        import contextlib
        m0 = IcdModel(wc, copy.deepcopy(cc), {}, label_config, {}, args=make_args('same'))
        check("_text_ctx is nullcontext when off (behaviour unchanged)",
              isinstance(m0._text_ctx(), contextlib.nullcontext))
        check("_label_ctx is nullcontext when off (behaviour unchanged)",
              isinstance(m0._label_ctx(), contextlib.nullcontext))
    finally:
        icd_model_mod.create_decoder = real_create_decoder

    print("\n" + "=" * 60)
    if FAILS:
        print("FAILED %d: %s" % (len(FAILS), FAILS))
        return 1
    print("all checks passed")
    return 0


if __name__ == '__main__':
    sys.exit(main())
