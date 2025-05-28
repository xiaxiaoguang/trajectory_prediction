from torch import nn
import torch
from abc import ABC
from module.utils import weight_init, top_n_accuracy
from module.subutils import seq2seq_forward,rnn_forward


class RnnLocPredictor(nn.Module, ABC):
    def __init__(self, embed_layer, input_size, rnn_hidden_size, fc_hidden_size, output_size, num_layers, seq2seq=True):
        super().__init__()
        self.__dict__.update(locals())

        self.encoder = nn.RNN(input_size, rnn_hidden_size, num_layers, dropout=0.1, batch_first=True)
        self.decoder = nn.RNN(input_size, rnn_hidden_size, num_layers, dropout=0.1, batch_first=True)
        self.out_linear = nn.Sequential(
            nn.Tanh(),
            nn.Linear(rnn_hidden_size, fc_hidden_size),
            nn.LeakyReLU(),
            nn.Linear(fc_hidden_size, output_size)
        )
        self.sos = nn.Parameter(torch.zeros(input_size).float(), requires_grad=True)
        self.apply(weight_init)

        self.embed_layer = embed_layer
        self.add_module('embed_layer', self.embed_layer)

    def forward(self, full_seq, valid_len, pre_len, **kwargs):
        """
        :param full_seq: input sequence tensor, shape (batch, seq_len)
        :param valid_len: length of valid sequence (non-padded), shape (batch,)
        :param pre_len: prediction horizon length
        :return: output predictions, shape (batch, pre_len, output_size)

        Explanation:
        The method first embeds the full input sequence using the embed_layer.
        Then, depending on whether seq2seq mode is enabled:
         - If seq2seq=True, it uses separate encoder and decoder RNNs with the seq2seq_forward function
           to generate predictions for the future steps.
         - Otherwise, it uses a single RNN and a start-of-sequence (sos) token with rnn_forward
           to predict the future sequence.
        Finally, the output from the RNN is passed through a feed-forward neural network to produce
        the final output with the desired dimensionality.
        """
        full_embed = self.embed_layer(full_seq, downstream=True, pre_len=pre_len, **kwargs)
        if self.seq2seq:
            rnn_out_pre = seq2seq_forward(self.encoder, self.decoder, full_embed, valid_len, pre_len)
        else:
            rnn_out_pre = rnn_forward(self.encoder, self.sos, full_embed, valid_len, pre_len)
        out = self.out_linear(rnn_out_pre)
        return out

