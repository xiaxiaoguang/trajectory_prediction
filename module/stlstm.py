from torch import nn
import torch
from abc import ABC
from module.utils import weight_init, top_n_accuracy
from module.subutils import seq2seq_forward,rnn_forward

class StlstmLocPredictor(nn.Module, ABC):
    def __init__(self, embed_layer, num_slots, aux_embed_size, time_thres, dist_thres,
                 input_size, lstm_hidden_size, fc_hidden_size, output_size, num_layers, seq2seq=True):
        """
        Spatio-temporal LSTM-based location predictor.

        @param embed_layer: Module that embeds event tokens into dense vectors.
        @param num_slots: Number of discretized bins for time/distance intervals.
        @param aux_embed_size: Dimensionality of auxiliary embeddings (time and distance).
        @param time_thres: Maximum value for clamping time differences.
        @param dist_thres: Maximum value for clamping spatial distances.
        @param input_size: Size of primary input embeddings.
        @param lstm_hidden_size: LSTM hidden state dimensionality.
        @param fc_hidden_size: Size of hidden layer in output MLP.
        @param output_size: Number of output classes or values.
        @param num_layers: Number of stacked LSTM layers.
        @param seq2seq: Whether to use seq2seq-style decoding or autoregressive decoding.
        """
        super().__init__()
        self.__dict__.update(locals())

        self.time_embed = nn.Embedding(num_slots + 1, aux_embed_size)
        self.dist_embed = nn.Embedding(num_slots + 1, aux_embed_size)

        self.encoder = nn.LSTM(input_size + 2 * aux_embed_size, lstm_hidden_size, num_layers, dropout=0.1, batch_first=True)
        self.decoder = nn.LSTM(input_size + 2 * aux_embed_size, lstm_hidden_size, num_layers, dropout=0.1, batch_first=True)

        self.out_linear = nn.Sequential(
            nn.Tanh(),
            nn.Linear(lstm_hidden_size, fc_hidden_size),
            nn.Tanh(),
            nn.Linear(fc_hidden_size, output_size)
        )

        self.sos = nn.Parameter(torch.zeros(input_size + 2 * aux_embed_size), requires_grad=True)
        self.aux_sos = nn.Parameter(torch.zeros(aux_embed_size * 2), requires_grad=True)

        self.apply(weight_init)

        self.embed_layer = embed_layer
        self.add_module('embed_layer', self.embed_layer)

    def forward(self, full_seq, valid_len, pre_len, **kwargs):
        """
        Forward pass to predict next event locations based on spatio-temporal context.

        @param full_seq: Tensor (batch_size, seq_len), token ids of events.
        @param valid_len: Tensor (batch_size,), lengths of valid history per sequence.
        @param pre_len: Integer, length of prediction horizon.
        @param kwargs: Must include 'time_delta' and 'dist' tensors for auxiliary features.

        @return: Tensor (batch_size, pre_len, output_size), predicted location probabilities or values.
        """
        batch_size = full_seq.size(0)

        # Step 1: Discretize time and distance deltas
        time_delta = kwargs['time_delta'][:, 1:]  # drop the first slot
        dist = kwargs['dist'][:, 1:]

        time_slot_i = torch.floor(torch.clamp(time_delta, 0, self.time_thres) / self.time_thres * self.num_slots).long()
        dist_slot_i = torch.floor(torch.clamp(dist, 0, self.dist_thres) / self.dist_thres * self.num_slots).long()

        # Step 2: Lookup embeddings for discretized auxiliary features
        time_emb = self.time_embed(time_slot_i)
        dist_emb = self.dist_embed(dist_slot_i)
        aux_input = torch.cat([
            self.aux_sos.view(1, 1, -1).expand(batch_size, 1, -1),
            torch.cat([time_emb, dist_emb], dim=-1)
        ], dim=1)

        # Step 3: Embed event input and concatenate with auxiliary input
        full_embed = self.embed_layer(full_seq, downstream=True, pre_len=pre_len, **kwargs)
        lstm_input = torch.cat([full_embed, aux_input], dim=-1)

        # Step 4: LSTM-based prediction
        if self.seq2seq:
            lstm_out_pre = seq2seq_forward(self.encoder, self.decoder, lstm_input, valid_len, pre_len)
        else:
            lstm_out_pre = rnn_forward(self.encoder, self.sos, lstm_input, valid_len, pre_len)

        out = self.out_linear(lstm_out_pre)
        return out
