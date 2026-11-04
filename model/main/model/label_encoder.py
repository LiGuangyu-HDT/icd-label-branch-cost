import torch
from torch import nn
from model.mlp import MLP


class LabelEncoder(nn.Module):
    def __init__(self, label_config={}):
        super(LabelEncoder, self).__init__()
        self.label_config = label_config
        self.pooling = self.label_config['pooling']
        # varlen: the input is ragged (SL, H), so pooling is per segment
        self.varlen = label_config.get('varlen', False)
        if label_config['num_layers'] > 0:
            self.mlp = MLP(input_dim=label_config['input_dim'],
                           hidden_dim=label_config['input_dim'],
                           output_dim=label_config['input_dim'],
                           num_layers=label_config['num_layers'],
                           dropout=label_config['dropout'])

    def _forward_varlen(self, input_feat, word_mask):
        """input_feat (SL, H), word_mask (n_labels, L_label)  ->  (n_labels, H).

        Matches the padded version: only real tokens are pooled. The original ``max`` pool
        did not mask padding, which is why the Transformer output used to be zeroed at those
        positions; on the ragged path there is no padding to worry about.
        """
        from model.varlen import (lengths_from_mask, lengths_to_cu_seqlens,
                                  segment_max, segment_mean)
        if word_mask is None:
            raise ValueError("the varlen LabelEncoder needs word_mask")
        lengths = lengths_from_mask(word_mask)
        cu = lengths_to_cu_seqlens(lengths)
        n_seg = int(lengths.numel())

        if not hasattr(self, 'pooling') or self.pooling == 'max':
            return segment_max(input_feat, cu, n_seg)
        if self.pooling == 'mean':
            return segment_mean(input_feat, cu, n_seg)
        if self.pooling == 'last':
            idx = (cu[1:] - 1).clamp(min=0).long()
            last = input_feat[idx]
            # A zero-length segment would make cu[i+1]-1 == cu[i]-1 point at the previous
            # segment's last token, so it is set to 0, as the padded path does.
            empty = (cu[1:] - cu[:-1]) == 0
            if bool(empty.any()):
                last = last.masked_fill(empty.unsqueeze(-1), 0.0)
            return last
        raise NotImplementedError(self.pooling)

    def forward(self, input_feat, word_mask=None):
        # input_feat: Label * Seq * Hidden
        if self.varlen:
            input_feat = self._forward_varlen(input_feat, word_mask)
        elif not hasattr(self, 'pooling') or self.pooling == 'max':
            input_feat = input_feat.max(dim=1)[0]  # Label * Hidden
        elif self.pooling == 'mean':
            input_feat = input_feat.mean(dim=1)
        elif self.pooling == 'last':
            assert word_mask is not None
            word_length = word_mask.sum(-1)
            word_length[word_length==0] = 1
            idx = (word_length - 1).unsqueeze(-1).unsqueeze(-1).repeat(1,1,input_feat.shape[-1])
            input_feat = torch.gather(input_feat, 1, idx).squeeze(1)
            
        if hasattr(self, 'mlp'):
            input_feat = self.mlp(input_feat)
        return input_feat
