from torch import nn
import torch
from abc import ABC
from module.utils import weight_init, top_n_accuracy
from module.subutils import seq2seq_forward,rnn_forward

class ErppLocPredictor(nn.Module, ABC):
    def __init__(self, embed_layer, input_size, lstm_hidden_size, fc_hidden_size, output_size, num_layers, seq2seq=True):
        """
        LSTM-based event location predictor using either standard RNN forward or seq2seq decoding.

        @param embed_layer: Module to embed input indices into vectors.
        @param input_size: Dimensionality of each input embedding.
        @param lstm_hidden_size: Hidden size of LSTM layers.
        @param fc_hidden_size: Size of the intermediate fully connected layer.
        @param output_size: Number of target classes or regression outputs.
        @param num_layers: Number of stacked LSTM layers.
        @param seq2seq: Whether to use seq2seq-style decoding or not.
        """
        super().__init__()
        self.__dict__.update(locals())

        self.encoder = nn.LSTM(input_size + 1, lstm_hidden_size, num_layers, dropout=0.1, batch_first=True)
        self.decoder = nn.LSTM(input_size + 1, lstm_hidden_size, num_layers, dropout=0.1, batch_first=True)

        self.mlp = nn.Sequential(
            nn.Tanh(),
            nn.Linear(lstm_hidden_size, fc_hidden_size),
            nn.Tanh()
        )
        self.event_linear = nn.Linear(fc_hidden_size, output_size)

        self.sos = nn.Parameter(torch.zeros(input_size + 1).float(), requires_grad=True)

        self.apply(weight_init)

        self.embed_layer = embed_layer
        self.add_module('embed_layer', self.embed_layer)

    def forward(self, full_seq, valid_len, pre_len, **kwargs):
        """
        Forward method for predicting event locations over a future time span.

        @param full_seq: Tensor of shape (batch_size, seq_len), combining history and target indices.
                         Includes zero padding and continuation after history.
        @param valid_len: Tensor (batch_size,) indicating the length of the observed sequence per sample.
        @param pre_len: Integer, number of prediction steps.
        @param kwargs: Additional inputs including timestamps (required: `timestamp` key).

        @return: Tensor of shape (batch_size, pre_len, output_size), predicted event location outputs.
        """
        event_embedding = self.embed_layer(full_seq, downstream=True, pre_len=pre_len, **kwargs)

        timestamp = kwargs['timestamp']
        hours = timestamp % (24 * 60 * 60) / 60 / 60 / 24  # Normal`ize time to fraction of a day
        lstm_input = torch.cat([event_embedding, hours.unsqueeze(-1)], dim=-1)

        if self.seq2seq:
            lstm_out_pre = seq2seq_forward(self.encoder, self.decoder, lstm_input, valid_len, pre_len)
        else:
            lstm_out_pre = rnn_forward(self.encoder, self.sos, lstm_input, valid_len, pre_len)

        mlp_out = self.mlp(lstm_out_pre)
        event_out = self.event_linear(mlp_out)
        return event_out

