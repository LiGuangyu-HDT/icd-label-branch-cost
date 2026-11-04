import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from reformer_pytorch import Reformer
from model.pos import AbsPositionEmbedding


class Combiner(nn.Module):
    def __init__(self, combine_config={}):
        super(Combiner, self).__init__()
        self.combine_config = combine_config
        self.input_dim = self.combine_config['input_dim']
        self.output_dim = self.combine_config['dim']
        
    def lstm_forward(self, x, lengths, lstm):
        np_lengths = lengths.cpu().numpy()
        np_lengths[np_lengths==0] = 1
        x_pack = pack_padded_sequence(x, np_lengths, batch_first=True, enforce_sorted=False)
        h_pack, _ = lstm(x_pack)
        h, _ = pad_packed_sequence(h_pack, batch_first=True)
        return h
    
    def combine_forward(self, embeds, word_mask, enc_repr=None):
        raise NotImplementedError

    def forward(self, word_hidden, word_mask, enc_repr=None):
        embeds = self.combine_forward(word_hidden, word_mask, enc_repr)
        if hasattr(self, 'reduce_linear'):
            embeds = self.reduce_linear(embeds)
        return embeds

class Naive_Combiner(Combiner):
    def __init__(self, combine_config={}):
        super(Naive_Combiner, self).__init__(combine_config)
        if self.input_dim != self.output_dim:
            self.reduce_linear = nn.Linear(self.input_dim, self.output_dim)
            
    def combine_forward(self, embeds, word_mask, enc_repr=None):
        return embeds
        
        
class LSTM_Combiner(Combiner):
    def __init__(self, combine_config={}):
        super(LSTM_Combiner, self).__init__(combine_config)
        self.rnn_dim = self.combine_config['rnn_dim']
        self.num_layers = self.combine_config['num_layers']
        self.combine_lstm_dropout = self.combine_config['lstm_dropout']
        self.combine_lstm = nn.LSTM(self.input_dim,
                                 self.rnn_dim // 2,
                                 self.num_layers,
                                 bidirectional=True,
                                 dropout=self.combine_lstm_dropout,
                                 batch_first=True)
        if self.rnn_dim != self.output_dim:
            self.reduce_linear = nn.Linear(self.rnn_dim, self.output_dim)
    
    def combine_forward(self, embeds, word_mask, enc_repr=None):
        word_lengths = torch.sum(word_mask, dim=1)
        h = self.lstm_forward(embeds, word_lengths, self.combine_lstm)
        return h
    
    
class Reformer_Combiner(Combiner):
    def __init__(self, combine_config={}):
        super(Reformer_Combiner, self).__init__(combine_config)

        self.rnn_dim = self.combine_config['rnn_dim']
        self.num_layers = self.combine_config['num_layers']
        self.reformer_head = self.combine_config['reformer_head']
        self.n_hashes = self.combine_config['n_hashes']
        self.linear_reformer = nn.Linear(self.input_dim, self.rnn_dim)
        self.pos_emb = AbsPositionEmbedding(self.combine_config)
        self.local_attention_head = self.combine_config['local_attention_head']
        self.pkm_layers = self.combine_config['pkm_layers']
        self.dropout = nn.Dropout(self.combine_config['transformer_dropout'])
        
        if self.combine_config['layer_norm']:
            self.norm = nn.LayerNorm(self.rnn_dim)
        
        self.reformer = Reformer(
            dim = self.rnn_dim,
            depth = self.num_layers,
            heads = self.reformer_head,
            n_hashes = self.n_hashes,
            lsh_dropout = 0.1,
            n_local_attn_heads = self.local_attention_head,
            pkm_layers = self.pkm_layers,
            causal = False)
        
        if self.rnn_dim != self.output_dim:
            self.reduce_linear = nn.Linear(self.rnn_dim, self.output_dim)

    def combine_forward(self, embeds, word_mask):
        x_embed = self.linear_reformer(embeds)
        x_embed = self.pos_emb(x_embed) + x_embed
        if self.combine_config['layer_norm']:
            x_embed = self.norm(x_embed)
        x_embed = self.dropout(x_embed)
        word_hidden = self.reformer(x_embed, i_mask=word_mask)
        return word_hidden  
        

class Transformer_Combiner(Combiner):
    """Drop-in replacement for the MSMN encoder: BiLSTM swapped for a standard Transformer.

    Differences from Reformer_Combiner:
      - Uses nn.TransformerEncoder, which goes through PyTorch's SDPA and therefore selects
        the Flash Attention kernel under bf16 without needing the flash-attn package.
      - Handles padding with src_key_padding_mask.
      - Zeroes the output at padded positions, matching BiLSTM's pad_packed_sequence.
        This must match: LabelEncoder's max-pool does not mask, so a non-zero value there
        would make the label branch behave differently from the baseline.
    """
    def __init__(self, combine_config={}):
        super(Transformer_Combiner, self).__init__(combine_config)
        self.rnn_dim = self.combine_config['rnn_dim']
        self.num_layers = self.combine_config['num_layers']
        self.n_head = self.combine_config['transformer_head']
        self.ff_dim = self.combine_config['transformer_ff']
        self.act = self.combine_config.get('transformer_activation', 'gelu')
        self.norm_first = self.combine_config.get('norm_first', False)
        self.zero_pad_output = self.combine_config.get('zero_pad_output', True)
        # varlen: ragged representation via flash-attn's variable-length interface, which
        # needs no padding mask. The output becomes (SL, rnn_dim), so downstream
        # (LabelEncoder, decoder) must enable varlen too.
        self.varlen = combine_config.get('varlen', False)
        # Encoder representation: padded / varlen (flash) / bucketed (grouped by exact
        # length, no padding). Defaults to varlen when the flag is on, else padded.
        self.enc_repr = combine_config.get('enc_repr') or ('varlen' if self.varlen else 'padded')
        # Chunk with one checkpoint per chunk (0: no chunking). Unchunked, the label
        # branch needs >24 GB for forward+backward; layer-level checkpointing still 13.73 GB.
        self.enc_chunk_tokens = int(combine_config.get('enc_chunk_tokens', 0) or 0)
        # SDPA backend for the dense path: 'efficient' is the explicit non-flash choice the
        # baseline cell needs; 'auto' lets PyTorch pick, which selects flash under bf16 with
        # no mask. Only the string is stored; the enum is not picklable.
        self.enc_attn_backend = (combine_config.get('enc_attn_backend') or 'efficient')

        self.linear_transformer = nn.Linear(self.input_dim, self.rnn_dim)
        self.dropout = nn.Dropout(self.combine_config['transformer_dropout'])
        if self.combine_config['layer_norm']:
            self.norm = nn.LayerNorm(self.rnn_dim)

        if self.varlen:
            from model.varlen import VarlenEncoderStack, VarlenPositionEmbedding
            self.pos_emb = VarlenPositionEmbedding(self.combine_config)
            self.transformer = VarlenEncoderStack(
                self.num_layers, self.rnn_dim, self.n_head, self.ff_dim,
                self.combine_config['transformer_dropout'], self.act, self.norm_first,
                dense_backend=self.enc_attn_backend)
        else:
            self.pos_emb = AbsPositionEmbedding(self.combine_config)
            layer = nn.TransformerEncoderLayer(
                d_model=self.rnn_dim,
                nhead=self.n_head,
                dim_feedforward=self.ff_dim,
                dropout=self.combine_config['transformer_dropout'],
                activation=self.act,
                batch_first=True,
                norm_first=self.norm_first)
            self.transformer = nn.TransformerEncoder(layer, num_layers=self.num_layers)

        if self.rnn_dim != self.output_dim:
            self.reduce_linear = nn.Linear(self.rnn_dim, self.output_dim)

    @staticmethod
    def _padding_mask(word_mask):
        """word_mask (1=real / 0=padding)  ->  src_key_padding_mask (True=ignore).

        A fully masked row makes softmax produce NaN, so one position is always kept open.
        """
        m = (word_mask == 0)
        if bool(m.all(dim=1).any()):
            m = m.clone()
            m[m.all(dim=1), 0] = False
        return m

    def combine_forward(self, embeds, word_mask, enc_repr=None):
        r = enc_repr or self.enc_repr
        if r != 'padded' or self.varlen:
            return self._combine_forward_ragged(embeds, word_mask, r)

        x = self.linear_transformer(embeds)
        x = self.pos_emb(x) + x
        if self.combine_config['layer_norm']:
            x = self.norm(x)
        x = self.dropout(x)

        pad_mask = self._padding_mask(word_mask) if word_mask is not None else None
        # The padded path supports chunking too; otherwise --enc_chunk_tokens would be
        # silently ignored and large batches would OOM.
        if self.enc_chunk_tokens and x.shape[1] and \
                x.shape[0] * x.shape[1] > self.enc_chunk_tokens:
            rows = max(1, self.enc_chunk_tokens // x.shape[1])
            outs = []
            for i in range(0, x.shape[0], rows):
                xc = x[i:i + rows].contiguous()
                pc = pad_mask[i:i + rows] if pad_mask is not None else None
                outs.append(torch.utils.checkpoint.checkpoint(
                    lambda t, pc=pc: self.transformer(t, src_key_padding_mask=pc),
                    xc, use_reentrant=False))
            word_hidden = torch.cat(outs, 0)
        else:
            word_hidden = self.transformer(x, src_key_padding_mask=pad_mask)

        if self.zero_pad_output and word_mask is not None:
            word_hidden = word_hidden * word_mask.unsqueeze(-1).to(word_hidden.dtype)
        return word_hidden

    def _combine_forward_ragged(self, embeds, word_mask, r):
        """Ragged path: concatenate to (SL, ...) and describe boundaries with cu_seqlens,
        so no padding mask is needed.

        ``r='varlen'``   flash-attn's variable-length interface (fp16/bf16)
        ``r='bucketed'`` group by exact length + dense SDPA (no waste, runs in fp32)
        ``r='padded'``   stack only; the text branch still uses padded + mask

        Returns (SL, rnn_dim) for varlen/bucketed and (B, L, rnn_dim) for padded.
        """
        from model.varlen import (lengths_from_mask, lengths_to_cu_seqlens, unpad)

        if word_mask is None:
            raise ValueError("the ragged path needs word_mask to derive true sequence lengths")
        if r not in ('varlen', 'bucketed', 'padded'):
            raise ValueError("unknown encoder representation %r (padded/varlen/bucketed)" % r)

        lengths = lengths_from_mask(word_mask)              # (B,)
        x = unpad(embeds, word_mask)                       # (ΣL, input_dim)
        x = self.linear_transformer(x)
        x = self.pos_emb(x, lengths) + x                   # within-segment positions
        if self.combine_config['layer_norm']:
            x = self.norm(x)
        x = self.dropout(x)

        if x.shape[0] == 0:
            return x

        if r == 'padded':
            # Text branch: lengths within a batch are similar so little is wasted, and the
            # decoder (naive/mem_efficient) needs (B, L, H), so re-pad here with zeros.
            L = int(word_mask.shape[1])
            xp = x.new_zeros(word_mask.shape[0], L, x.shape[-1])
            xp = xp.index_put((word_mask.bool(),), x)
            return self.transformer.forward_dense_pad(
                xp, word_mask, self.enc_chunk_tokens)

        if r == 'bucketed':
            return self.transformer.forward_bucketed(
                x, lengths, self.enc_chunk_tokens or 1 << 30)

        cu = lengths_to_cu_seqlens(lengths)
        max_len = int(lengths.max().item())
        if self.enc_chunk_tokens:
            return self.transformer.forward_chunked(x, cu, self.enc_chunk_tokens)
        return self.transformer(x, cu, max_len)


def create_combiner(combine_config):
    if combine_config['model'] == 'naive':
        combiner = Naive_Combiner(combine_config)
    if combine_config['model'] == 'lstm':
        combiner = LSTM_Combiner(combine_config)
    if combine_config['model'] == 'reformer':
        combiner = Reformer_Combiner(combine_config)
    if combine_config['model'] == 'transformer':
        combiner = Transformer_Combiner(combine_config)
    return combiner
