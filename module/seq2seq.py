from torch import nn
from abc import ABC
from module.utils import weight_init, top_n_accuracy
from module.subutils import seq2seq_forward

class Seq2SeqLocPredictor(nn.Module, ABC):
    """
    A sequence-to-sequence model for next location prediction, built using GRUs.
    """

    def __init__(self, embed_layer, input_size, hidden_size, output_size, num_layers):
        super().__init__()
        self.__dict__.update(locals())  # Stores constructor arguments as instance attributes (not recommended for all use cases)

        # Encoder GRU to process historical sequence
        self.encoder = nn.GRU(input_size, hidden_size, num_layers, dropout=0.1, batch_first=True)

        # Decoder GRU to generate future locations
        self.decoder = nn.GRU(input_size, hidden_size, num_layers, dropout=0.1, batch_first=True)

        # Output projection layer: maps decoder hidden state to output space
        self.out_linear = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),  # Expand hidden dim
            nn.LeakyReLU(),                           # Non-linear activation
            nn.Dropout(0.1),                           # Regularization
            nn.Linear(hidden_size * 4, output_size)    # Project to output size
        )

        self.apply(weight_init)  # Apply custom weight initialization

        self.embed_layer = embed_layer  # Embedding layer for input sequence
        self.add_module('embed_layer', self.embed_layer)  # Register embed_layer properly as submodule

    def forward(self, full_seq, valid_len, pre_len, **kwargs):
        """
        Forward pass of the model.

        @param full_seq: Tensor of shape (batch_size, seq_len) with historical and target sequence indices.
                         Each row follows format: [l_1, ..., l_h, 0, 0, 0, l_h+1, ..., l_h+n]
                         where h is the length of the history and n is the length of prediction.
        @param valid_len: 1D tensor (batch_size,) indicating valid length of each full_seq in the batch.
        @param pre_len:   Scalar indicating how many future steps to predict.
        """
        # Embed the input sequences; returns (batch_size, seq_len, input_size)
        lstm_input = self.embed_layer(full_seq, downstream=True, pre_len=pre_len, **kwargs)

        # Use encoder and decoder GRUs in a seq2seq fashion
        decoder_out = seq2seq_forward(self.encoder, self.decoder, lstm_input, valid_len, pre_len)

        # Final linear projection to prediction space (e.g., logits over locations)
        out = self.out_linear(decoder_out)  # Shape: (batch_size, pre_len, output_size)
        return out

