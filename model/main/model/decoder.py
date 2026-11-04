import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from preprocess.coder_init_code_embedding import coder_init, load_code_embedding
from model.mlp import MLP
from torch.nn.init import xavier_uniform_ as xavier_uniform
from opt_einsum import contract

# The varlen path (attn_impl='varlen') needs flash-attn; other paths work without it
try:
    from flash_attn import flash_attn_varlen_func
except Exception:                                    # pragma: no cover
    flash_attn_varlen_func = None


class Decoder(nn.Module):
    def __init__(self, decoder_config={}):
        super(Decoder, self).__init__()
        self.decoder_config = decoder_config
        self.input_dim = self.decoder_config['input_dim']
        self.attention_dim = self.decoder_config['attention_dim']
        self.code_embedding_path = self.decoder_config['code_embedding_path']
        self.ind2c = self.decoder_config['ind2c']
        self.ind2mc = self.decoder_config['ind2mc']
        
        self.final = nn.Linear(self.input_dim, len(self.ind2c))
        self.ignore_mask = False
        
        self.est_cls = self.decoder_config['est_cls']
        if self.est_cls > 0:
            self.w_linear = MLP(self.attention_dim, self.attention_dim, self.attention_dim, self.est_cls)
            self.b_linear = MLP(self.attention_dim, self.attention_dim, 1, self.est_cls)
        elif self.est_cls == -1:
            self.w_linear = nn.Identity()
            self.b_linear = self.zero_first_dim

    def zero_first_dim(self, x):
        return torch.zeros_like(x)[:,0]

    def forward(self, h, word_mask, label_feat=None):
        m = self.get_label_queried_features(h, word_mask, label_feat)
        
        if hasattr(self, 'w_linear'):
            w = self.w_linear(label_feat) # label * hidden
            b = self.b_linear(label_feat) # label * 1
            logits = self.get_logits(m, w, b)
        else:
            logits = self.get_logits(m)
        return logits
    
    def get_logits(self, m, w=None, b=None):
        # m: batch * label * hidden
        if w is None:
            logits = self.final.weight.mul(m).sum(dim=2).add(self.final.bias)
        else:
            logits = contract('blh,lh->bl', m, w) + b.squeeze(-1)
        return logits
    
    def get_label_queried_features(self, h, word_mask, label_feat):
        raise NotImplementedError
    
    def _code_emb_init(self, layer, version=None, init_code='ind2c'):
        if not self.code_embedding_path:
            return
        dim = layer.weight.shape[1]
        version = "mimic3"

        try:
            if init_code == "ind2c":
                ind2c = self.ind2c
            if init_code == "ind2mc":
                ind2c = self.ind2mc
            label_count = len(ind2c)
            code_embs = coder_init(self.code_embedding_path, ind2c, dim, 'cuda:0', version)
        except BaseException:
            code_embs = load_code_embedding(self.code_embedding_path)

        print(f'Init Layer {layer} using {self.code_embedding_path}')
            
        weights = np.zeros(layer.weight.size())
        for i in range(label_count):
            code = ind2c[i]
            weights[i] = code_embs[code]
        layer.weight.data[0:weights.shape[0]] = torch.Tensor(weights).clone()


class LAAT(Decoder):
    def __init__(self, decoder_config={}):
        super(LAAT, self).__init__(decoder_config)
        self.W = nn.Linear(self.input_dim, self.attention_dim)
        self.U = nn.Linear(self.attention_dim, len(self.ind2mc))
        
        self.xavier = self.decoder_config['xavier']
        
        if self.xavier:
            xavier_uniform(self.W.weight)
        self._code_emb_init(self.U, init_code='ind2mc')
    
    def get_label_queried_features(self, h, word_mask=None, label_feat=None):
        if word_mask is not None and not self.ignore_mask:
            l = word_mask.shape[-1]
            h = h[:,0:l]
        z = torch.tanh(self.W(h))
        if label_feat is None:
            label_feat = self.U.weight
        score = label_feat.matmul(z.transpose(1,2))
        if word_mask is not None and not self.ignore_mask:
            word_mask = word_mask.bool()
            score = score.masked_fill(mask=~word_mask[:,0:score.shape[-1]].unsqueeze(1).expand_as(score), value=float('-1e6'))
        alpha = F.softmax(score, dim=2)
        m = alpha.matmul(h) # Batch * Label * Hidden
        return m
    
 
class MultiLabelMultiHeadLAAT(LAAT):
    def __init__(self, decoder_config={}):
        super(MultiLabelMultiHeadLAAT, self).__init__(decoder_config)
        self.attention_head = self.decoder_config['attention_head']
        #assert self.input_dim % self.attention_head == 0
        #assert self.attention_dim % self.attention_head == 0
        self.rep_dropout = nn.Dropout(decoder_config['rep_dropout'])
        
        self.head_pooling = decoder_config['head_pooling']
        if self.head_pooling == "concat":
            assert self.attention_dim % self.attention_head == 0
            self.reduce = nn.Linear(self.attention_dim,
                                     self.attention_dim // self.attention_head)
        if decoder_config.get('att_dropout') > 0.0:
            self.att_dropout_rate = decoder_config['att_dropout']
            self.att_dropout = nn.Dropout(self.att_dropout_rate)
        
    def get_label_queried_features(self, h, word_mask=None, label_feat=None):
        if word_mask is not None:
            if not hasattr(self, 'ignore_mask') or not self.ignore_mask:
                l = word_mask.shape[-1]
                h = h[:,0:l]
        z = torch.tanh(self.W(h)) # batch_size * seq_length * att_dim
        batch_size, seq_length, att_dim = z.size()
        # batch_size, seq_length, att_head, sub_dim
        if label_feat is None:
            label_feat = self.U.weight
        label_count = label_feat.size(0) // self.attention_head
        u_reshape = label_feat.reshape(label_count, self.attention_head, att_dim)
        # label_count * att_head * dim
        score = contract('abd,ecd->aebc', z, u_reshape)
        # batch_size, label_count, seq_length, att_head
        
        if word_mask is not None:
            if not hasattr(self, 'ignore_mask') or not self.ignore_mask:
                word_mask = word_mask.bool()
                score = score.masked_fill(mask=~word_mask[:,0:score.shape[-2]].unsqueeze(1).unsqueeze(-1).expand_as(score),
                                        value=float('-1e6'))
        alpha = F.softmax(score, dim=2) # softmax on seq_length # batch_size, label_count, seq_length, att_head
        if hasattr(self, 'att_dropout'):
            alpha = self.att_dropout(alpha)
            if self.training:
                alpha_sum = torch.clamp(alpha.sum(dim=2, keepdim=True), 1e-5)
                alpha = alpha / alpha_sum
        # # batch_size, attention_head, label_count, seq_length
        
        # # batch_size, seq_length, attention_head, hidden // attention_head
        # # batch_size, attention_head, seq_length, hidden // attention_head
        m = contract('abd,aebc->aedc', h, alpha)# .reshape(batch_size, label_count, -1)
        # h: batch * seq * hidden
        # a: batch * label * seq * head
        # test max pooling here / can be changed to mlp/concat/attention
        if not hasattr(self, 'head_pooling') or self.head_pooling == "max":
            m = m.max(-1)[0]
        elif self.head_pooling == "concat":
            m = self.reduce(m.permute(0,1,3,2)) # batch * label * hidden // head * head
            m = m.reshape(batch_size, -1, att_dim)
        
        # batch_size, attention_head, label_count, hidden // attention_head
        # batch_size, label_count, attention_head, hidden // attention_head
        m = self.rep_dropout(m)
        return m
    
    def transform_label_feats(self, label_feat):
        if not hasattr(self, 'head_pooling') or self.head_pooling == "max":
            label_count = label_feat.shape[0] // self.attention_head
            label_feat = label_feat.reshape(label_count, self.attention_head, -1).max(1)[0]
        elif self.head_pooling == "concat":
            label_count = label_feat.shape[0] // self.attention_head
            label_feat = self.reduce(label_feat) # (label * head) * (hidden // head)
            label_feat = label_feat.reshape(label_count, self.attention_head, -1) # label * head * hidden
            label_feat = label_feat.reshape(label_count, -1)
        return label_feat

    def forward(self, h, word_mask, label_feat=None):
        m = self.get_label_queried_features(h, word_mask, label_feat)
        
        if hasattr(self, 'w_linear'):
            label_feat = self.transform_label_feats(label_feat)
            w = self.w_linear(label_feat) # label * hidden
            b = self.b_linear(label_feat) # label * 1
            logits = self.get_logits(m, w, b)
        else:
            logits = self.get_logits(m)
        return logits
    
ACT2FN = {'tanh':torch.tanh,
          'relu':torch.relu}
    
class MultiLabelMultiHeadLAATV2(MultiLabelMultiHeadLAAT):
    def __init__(self, decoder_config={}):
        super(MultiLabelMultiHeadLAATV2, self).__init__(decoder_config)
        self.act_fn_name = decoder_config['act_fn_name']
        self.act_fn = ACT2FN[self.act_fn_name]
        self.u_reduce = nn.Linear(self.attention_dim,
                                  self.attention_dim // self.attention_head)
        
    def _attention_memory_efficient(self, h, z, label_feat, word_mask, label_count,
                                    batch_size, seq_length, att_dim, attn_impl):
        """Variant 1c: map MSMN's multi-synonym attention onto memory-efficient / Flash
        Attention.
            q = u_reshape (C, M, d)      k = z_reshape (B, L, M, d)      v = h (B, L, 512)

        v's head width (att_dim=512) differs from q/k's (d=128) while SDPA requires all
        three to match, so q/k are zero-padded to vchunk and v is sliced along the head
        axis into att_dim/vchunk blocks. Padding with zeros leaves the dot product
        unchanged, so this is exact; the cost is recomputing q.k per block, making
        attention FLOPs 2M/(M+1) (1.60x at M=4).

        MSMN's attention has no scaling, so scale=1.0 must be passed explicitly; SDPA
        defaults to 1/sqrt(E) and silently changes the model otherwise.

        Returns (B, C, att_dim, M), matching the naive path so head_pooling is unchanged.
        """
        from torch.nn.attention import SDPBackend, sdpa_kernel

        head_dim = att_dim // self.attention_head
        vchunk = int(self.decoder_config.get('attn_vchunk', 256))
        if vchunk < head_dim:
            raise ValueError(
                "the 1c path (attn_impl=%s) requires attn_vchunk(%d) >= head_dim(%d)"
                % (attn_impl, vchunk, head_dim))
        if att_dim % vchunk != 0:
            raise ValueError(
                "the 1c path (attn_impl=%s) requires att_dim(%d) divisible by attn_vchunk(%d)"
                % (attn_impl, att_dim, vchunk))
        if hasattr(self, 'att_dropout'):
            raise NotImplementedError(
                "att_dropout cannot take effect under attn_impl=%s (dropout lives inside "
                "the SDPA kernel, and the original rescales after it). Set --att_dropout 0 "
                "or use --attn_impl naive." % attn_impl)

        # PyTorch's FLASH_ATTENTION backend accepts attn_mask=None only -- any mask, bool
        # or additive, raises "No available kernel" -- but the text here needs a padding
        # mask, so that backend is unusable. Only flash-attn's variable-length interface
        # (cu_seqlens instead of a mask) actually reaches Flash Attention on this shape.
        if attn_impl == 'flash' and word_mask is not None:
            raise RuntimeError(
                "attn_impl=flash is incompatible with a padding mask: PyTorch's SDPA flash "
                "backend accepts attn_mask=None only.\n"
                "       Options: (a) --attn_impl mem_efficient, which supports masks on the "
                "same memory-efficient path;\n"
                "                (b) flash-attn's variable-length interface, which needs "
                "the encoder/decoder reimplemented.")

        z_r = z.reshape(batch_size, seq_length, self.attention_head, head_dim)
        u_r = self.u_reduce(label_feat.reshape(label_count, self.attention_head, att_dim))

        # Zero-pad q/k to vchunk and move to SDPA's (B, M, *, E) layout
        q = F.pad(u_r.unsqueeze(0).expand(batch_size, -1, -1, -1), (0, vchunk - head_dim))
        k = F.pad(z_r, (0, vchunk - head_dim))
        q = q.permute(0, 2, 1, 3)                     # (B, M, C, vchunk)
        k = k.permute(0, 2, 1, 3)                     # (B, M, L, vchunk)

        attn_mask = None
        if word_mask is not None:
            if not hasattr(self, 'ignore_mask') or not self.ignore_mask:
                # Same additive mask as the naive path (the original fills with -1e6)
                key_pad = (word_mask[:, 0:seq_length] == 0)
                attn_mask = torch.zeros(batch_size, 1, 1, seq_length,
                                        device=h.device, dtype=h.dtype)
                attn_mask = attn_mask.masked_fill(
                    key_pad[:, None, None, :], torch.tensor(-1e6, device=h.device, dtype=h.dtype))

        backend = {"mem_efficient": SDPBackend.EFFICIENT_ATTENTION,
                   "flash": SDPBackend.FLASH_ATTENTION}[attn_impl]
        outs = []
        with sdpa_kernel(backend):
            for j in range(att_dim // vchunk):
                v = h[..., j * vchunk:(j + 1) * vchunk]                   # (B, L, vchunk)
                v = v.unsqueeze(2).expand(-1, -1, self.attention_head, -1)
                v = v.permute(0, 2, 1, 3)                                 # (B, M, L, vchunk)
                outs.append(F.scaled_dot_product_attention(
                    q, k, v, attn_mask=attn_mask, scale=1.0))

        # (B, M, C, att_dim) → (B, C, att_dim, M)
        return torch.cat(outs, dim=-1).permute(0, 2, 3, 1)

    def _attention_varlen(self, h, word_mask, label_feat):
        """Variant 1c with flash-attn's variable-length kernel: cu_seqlens describes the
        ragged text, so no attention mask is needed.

        q holds codes (exactly C per sample, regular) while k/v are ragged text;
        ``flash_attn_varlen_func`` allows the two sides to use different cu_seqlens.

        The 1c zero-padding is still needed (flash-attn 2.8.3 has no headdim_v parameter).
        MSMN's attention has no scaling, so ``softmax_scale=1.0`` must be passed.

        Returns (m, batch_size, att_dim) with m of shape (B, C, att_dim, M), as elsewhere.
        """
        from model.varlen import (lengths_from_mask, lengths_to_cu_seqlens,
                                  require_flash_attn, _align_flash_dtype)
        require_flash_attn()
        if word_mask is None:
            raise ValueError("attn_impl=varlen needs word_mask to derive the true sequence lengths")
        if hasattr(self, 'att_dropout'):
            raise NotImplementedError(
                "att_dropout cannot take effect under attn_impl=varlen; set --att_dropout 0.")

        att_dim = self.attention_dim
        head_dim = att_dim // self.attention_head
        vchunk = int(self.decoder_config.get('attn_vchunk', 256))
        if vchunk < head_dim or att_dim % vchunk != 0:
            raise ValueError("the varlen 1c path requires attn_vchunk>=head_dim and att_dim%%vchunk==0")

        batch_size = word_mask.shape[0]
        lengths = lengths_from_mask(word_mask)
        cu_k = lengths_to_cu_seqlens(lengths)
        max_k = int(lengths.max().item())
        label_count = label_feat.size(0) // self.attention_head
        cu_q = torch.arange(0, (batch_size + 1) * label_count, label_count,
                            dtype=torch.int32, device=h.device)

        z = self.act_fn(self.W(h))                                   # (ΣL, att_dim)
        z_r = z.reshape(-1, self.attention_head, head_dim)           # (ΣL, M, d)
        u_r = self.u_reduce(label_feat.reshape(label_count, self.attention_head, att_dim))

        # 1c: zero-pad q/k to vchunk (zeros leave the dot product unchanged)
        q = F.pad(u_r.unsqueeze(0).expand(batch_size, -1, -1, -1), (0, vchunk - head_dim))
        q = q.reshape(batch_size * label_count, self.attention_head, vchunk).contiguous()
        k = F.pad(z_r, (0, vchunk - head_dim)).contiguous()          # (ΣL, M, vchunk)

        outs = []
        for j in range(att_dim // vchunk):
            v = h[..., j * vchunk:(j + 1) * vchunk]                  # (ΣL, vchunk)
            v = v.unsqueeze(1).expand(-1, self.attention_head, -1).contiguous()
            q2, k2, v2 = _align_flash_dtype(q, k, v)
            outs.append(flash_attn_varlen_func(
                q2, k2, v2, cu_q, cu_k, label_count, max_k,
                dropout_p=0.0, softmax_scale=1.0))
        out = torch.cat(outs, dim=-1)                                # (B*C, M, att_dim)
        m = out.reshape(batch_size, label_count, self.attention_head, att_dim).permute(0, 1, 3, 2)
        return m, batch_size, att_dim

    def _pool_heads(self, m, batch_size, att_dim):
        """Multi-synonym head pooling (max / concat), shared by the chunked and unchunked
        paths so that they agree."""
        if not hasattr(self, 'head_pooling') or self.head_pooling == "max":
            return m.max(-1)[0]
        if self.head_pooling == "concat":
            m = self.reduce(m.permute(0, 1, 3, 2))   # batch * label * hidden//head * head
            return m.reshape(batch_size, -1, att_dim)
        return m

    def _naive_core(self, z_reshape, u_chunk, h, pad_mask):
        """Naive attention for one code block: score, softmax, aggregate, head-pool.

        Pooling is folded in here because the point of chunking is to never materialise the
        (B, C, L, M) tensor, which would need 4.57 GB in fp32 at B=8, C=8921, L=4000, M=4.
        """
        score = contract('abcd,ecd->aebc', z_reshape, u_chunk)
        if pad_mask is not None:
            score = score.masked_fill(mask=pad_mask, value=float('-1e6'))
        alpha = F.softmax(score, dim=2)
        if hasattr(self, 'att_dropout'):
            alpha = self.att_dropout(alpha)
            if self.training:
                alpha *= (1 - self.att_dropout_rate)
        m = contract('abd,aebc->aedc', h, alpha)
        return self._pool_heads(m, z_reshape.shape[0], self.attention_dim)

    def get_label_queried_features(self, h, word_mask=None, label_feat=None):
        attn_impl = self.decoder_config.get('attn_impl', 'naive')
        pooled = False

        if attn_impl == 'varlen':
            # h is ragged (SL, att_dim); lengths come from word_mask
            if label_feat is None:
                label_feat = self.U.weight
            m, batch_size, att_dim = self._attention_varlen(h, word_mask, label_feat)
        else:
            if word_mask is not None:
                if not hasattr(self, 'ignore_mask') or not self.ignore_mask:
                    l = word_mask.shape[-1]
                    h = h[:,0:l]
            z = self.act_fn(self.W(h)) # batch_size * seq_length * att_dim
            batch_size, seq_length, att_dim = z.size()
            if label_feat is None:
                label_feat = self.U.weight
            label_count = label_feat.size(0) // self.attention_head

            if attn_impl != 'naive':
                m = self._attention_memory_efficient(
                    h, z, label_feat, word_mask, label_count,
                    batch_size, seq_length, att_dim, attn_impl)
            else:
                z_reshape = z.reshape(batch_size, seq_length, self.attention_head, att_dim // self.attention_head)
                # batch_size, seq_length, att_head, sub_dim
                u_reshape = self.u_reduce(label_feat.reshape(label_count, self.attention_head, att_dim))
                pad_mask = None
                if word_mask is not None:
                    if not hasattr(self, 'ignore_mask') or not self.ignore_mask:
                        # (B,1,L,1) broadcast to (B, dc, L, M) by masked_fill
                        pad_mask = (~word_mask.bool()[:, 0:seq_length]).unsqueeze(1).unsqueeze(-1)

                dc = int(self.decoder_config.get('dec_chunk_codes', 0) or 0)
                if dc and dc < label_count:
                    # Chunk along the code axis, one checkpoint per chunk. Attention is
                    # independent per code, so this is exact (fp32, B=8: 24.12 -> 4.22 GB
                    # for +3.6% time).
                    parts = []
                    for c0 in range(0, label_count, dc):
                        parts.append(torch.utils.checkpoint.checkpoint(
                            self._naive_core, z_reshape, u_reshape[c0:c0 + dc],
                            h, pad_mask, use_reentrant=False))
                    m = torch.cat(parts, dim=1)
                    pooled = True                      # already pooled inside the chunk
                else:
                    m = self._naive_core(z_reshape, u_reshape, h, pad_mask)
                    pooled = True                      # _naive_core already head-pools

        if not pooled:
            m = self._pool_heads(m, batch_size, att_dim)

        # batch_size, label_count, attention_head, hidden // attention_head
        m = self.rep_dropout(m)
        return m

def create_decoder(decoder_config):
    if decoder_config['model'] == 'LAAT':
        decoder = LAAT(decoder_config)
    if decoder_config['model'] == 'MultiLabelMultiHeadLAAT':
        decoder = MultiLabelMultiHeadLAAT(decoder_config)
    if decoder_config['model'] == 'MultiLabelMultiHeadLAATV2':
        decoder = MultiLabelMultiHeadLAATV2(decoder_config)
    return decoder
