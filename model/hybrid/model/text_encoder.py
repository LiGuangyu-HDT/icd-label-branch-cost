from torch import nn
from model.word_encoder import Word_Encoder
from model.combiner import create_combiner


class TextEncoder(nn.Module):
    def __init__(self, word_config={}, combine_config={}, word_encoder=None):
        super(TextEncoder, self).__init__()
        self.word_config = word_config
        self.combine_config = combine_config
        
        self.input_dim = 0
        # Hybrid combiner (--label_combiner lstm) may reuse an existing Word_Encoder for the
        # second branch instead of building another one, so that swapping the combiner does
        # not also change what the vocabulary and embeddings learn, and so that the extra
        # parameters come from the combiner alone. None builds a fresh one, as before.
        if word_encoder is None:
            self.word_encoder = Word_Encoder(self.word_config)
        else:
            self.word_encoder = word_encoder
        self.input_dim += self.word_config['dim']
        self.combine_config['input_dim'] = self.input_dim
        self.combiner = create_combiner(self.combine_config)
        
    def forward(self, input_word, word_mask, enc_repr=None):
        word_hidden = self.word_encoder(input_word)
        hidden = self.combiner(word_hidden, word_mask, enc_repr)
        return hidden
