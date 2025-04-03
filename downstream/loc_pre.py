from abc import ABC

import numpy as np
import torch
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from sklearn.utils import shuffle
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from tqdm import tqdm
from utils import next_batch, create_src_trg, weight_init, top_n_accuracy
import os 
import shutil
import datetime

def seq2seq_forward(encoder, decoder, lstm_input, valid_len, pre_len):
    his_len = (valid_len - pre_len).to('cpu')
    src_padded_embed = pack_padded_sequence(lstm_input, his_len, batch_first=True, enforce_sorted=False)
    _, hc = encoder(src_padded_embed)
    trg_embed = torch.stack([torch.cat([lstm_input[i, start - 1:start], lstm_input[i, -pre_len:-1]], dim=0)
                             for i, start in enumerate(his_len)], dim=0)
    decoder_out, _ = decoder(trg_embed, hc)  # (batch_size, pre_len, hidden_size)
    return decoder_out


def rnn_forward(encoder, sos, lstm_input, valid_len, pre_len):
    batch_size = lstm_input.size(0)
    input_size = lstm_input.size(-1)
    history_len = valid_len - pre_len
    max_len = history_len.max()

    lstm_input = torch.cat([sos.reshape(1, 1, -1).repeat(batch_size, 1, 1), lstm_input], dim=1)  # (batch, seq_len+1, 1+input_size)
    lstm_input = torch.stack([torch.cat([lstm_input[i, :s + 1], lstm_input[i, -pre_len:-1],
                                         torch.zeros(max_len - s, input_size).float().to(lstm_input.device)], dim=0)
                              for i, s in enumerate(history_len)], dim=0)
    lstm_out, _ = encoder(lstm_input)
    lstm_out_pre = torch.stack([lstm_out[i, s - pre_len:s] for i, s in enumerate(valid_len)])  # (batch, pre_len, lstm_hidden_size)
    return lstm_out_pre

class Seq2SeqLocPredictor(nn.Module, ABC):
    """
    A next location predictor constructed of LSTM.
    """
    def __init__(self, embed_layer, input_size, hidden_size, output_size, num_layers):
        super().__init__()
        self.__dict__.update(locals())

        self.encoder = nn.GRU(input_size, hidden_size, num_layers, dropout=0.1, batch_first=True)
        self.decoder = nn.GRU(input_size, hidden_size, num_layers, dropout=0.1, batch_first=True)
        self.out_linear = nn.Sequential(nn.Linear(hidden_size, hidden_size * 4), nn.LeakyReLU(),
                                        nn.Dropout(0.1), nn.Linear(hidden_size * 4, output_size))
        self.apply(weight_init)

        self.embed_layer = embed_layer
        self.add_module('embed_layer', self.embed_layer)

    def forward(self, full_seq, valid_len, pre_len, **kwargs):
        """
        @param full_seq: combined historical and target sequence, shape (batch_size, seq_len).
            One row: [l_1, l_2, ..., l_h, 0, 0, 0, l_h+1, ..., l_h+n], where h is the valid length of history sequence,
            n is the length of prediction sequence.
        @param valid_len: an 1D tensor carrying the legit length of full sequence for every batch, shape (batch_size)
        @param pre_len: a scalar indicates the length of prediction.
        """
        lstm_input = self.embed_layer(full_seq, downstream=True, pre_len=pre_len, **kwargs)  # (batch_size, seq_len, input_size)
        decoder_out = seq2seq_forward(self.encoder, self.decoder, lstm_input, valid_len, pre_len)
        out = self.out_linear(decoder_out)  # (batch_size, pre_len, output_size)
        return out

def seq2seq_forward(encoder, decoder, lstm_input, valid_len, pre_len):
    his_len = (valid_len - pre_len).to('cpu')
    src_padded_embed = pack_padded_sequence(lstm_input, his_len, batch_first=True, enforce_sorted=False)
    _, hc = encoder(src_padded_embed)
    trg_embed = torch.stack([torch.cat([lstm_input[i, start - 1:start], lstm_input[i, -pre_len:-1]], dim=0)
                             for i, start in enumerate(his_len)], dim=0)
    decoder_out, _ = decoder(trg_embed, hc)  # (batch_size, pre_len, hidden_size)
    return decoder_out

class TransformerPredictor(nn.Module, ABC):
    def __init__(self, embed_layer, input_size, hidden_size, output_size, num_layers, num_heads=8, ff_dim=512):
        super().__init__()
        self.__dict__.update(locals())        
        self.pre_len = None  # Will be set in forward
        self.embed_layer = embed_layer
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dim_feedforward=ff_dim, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=num_heads, dim_feedforward=ff_dim, dropout=0.1, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output layer
        self.out_linear = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 4, output_size)
        )

    def forward(self, full_seq, valid_len, pre_len, **kwargs):
        """
        @param full_seq: Input sequence (batch_size, seq_len)
        @param valid_len: Valid lengths (batch_size,)
        @param pre_len: Prediction length
        """
        self.pre_len = pre_len

        embedded_input = self.embed_layer(full_seq, downstream=True, pre_len=pre_len, **kwargs)  # (batch_size, seq_len, hidden_size)
        
        # Create masks (optional: for ignoring padding in attention)
        src_key_padding_mask = self._generate_padding_mask(full_seq, valid_len)
        tgt_mask = self._generate_square_subsequent_mask(pre_len)
        # Encoder processing
        memory = self.encoder(embedded_input, src_key_padding_mask=src_key_padding_mask)
        
        # Prepare decoder input (zero vector as start tokens)
        batch_size = full_seq.shape[0]
        decoder_input = torch.zeros((batch_size, pre_len, self.hidden_size), device=full_seq.device)
        
        # Decode
        decoder_out = self.decoder(decoder_input, memory, tgt_mask=tgt_mask, memory_key_padding_mask=src_key_padding_mask)
        
        # Output transformation
        out = self.out_linear(decoder_out)  # (batch_size, pre_len, output_size)
        return out

    def _generate_padding_mask(self, seq, valid_len):
        """ Creates a mask for padding positions based on valid lengths """
        batch_size, seq_len = seq.shape[:2]
        mask = torch.arange(seq_len, device=seq.device).expand(batch_size, seq_len) >= valid_len.unsqueeze(1)
        return mask  # Shape: (batch_size, seq_len)

    def _generate_square_subsequent_mask(self, size):
        """ Generates a lower triangular mask for the decoder """
        return torch.triu(torch.ones(size, size, device='cpu') * float('-inf'), diagonal=1).to(dtype=torch.float32)

class DecoderPredictor(nn.Module, ABC):
    def __init__(self, embed_layer, input_size, hidden_size, output_size, num_layers, num_heads=8, ff_dim=512):
        super().__init__()
        self.__dict__.update(locals())        
        
        self.embed_layer = embed_layer
        self.hidden_size = hidden_size

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads, dim_feedforward=ff_dim, dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.out_linear = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 4, output_size)
        )
        
    def forward(self, full_seq, valid_len, pre_len, **kwargs):
        self.pre_len = pre_len
        # embedded_input = self.embed_layer(full_seq) 
        embedded_input = self.embed_layer(full_seq, downstream=True, pre_len=pre_len, **kwargs)  # (batch_size, seq_len, hidden_size)

        src_key_padding_mask = self._generate_padding_mask(full_seq, valid_len)
        memory = self.encoder(embedded_input, src_key_padding_mask=src_key_padding_mask)
        out = self.out_linear(memory[:,:-pre_len,:]) 
        return out[:, -pre_len:, :]
    
    def _generate_padding_mask(self, seq, valid_len):
        batch_size, seq_len = seq.shape[:2]
        mask = torch.arange(seq_len, device=seq.device).expand(batch_size, seq_len) >= valid_len.unsqueeze(1)
        return mask  # Shape: (batch_size, seq_len)

    def _generate_causal_mask(self, seq_len, pre_len, device):
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)  # Upper triangular matrix
        mask[seq_len - pre_len:] = float('-inf')
        return mask  # Shape: (seq_len, seq_len)


class DecoderPredictor2(nn.Module, ABC):
    def __init__(self, embed_layer, input_size, hidden_size, output_size, num_layers, num_heads=8, ff_dim=512):
        super().__init__()
        self.__dict__.update(locals())        
        
        # Store parameters
        self.embed_layer = embed_layer
        self.hidden_size = hidden_size
        
        # Transformer Encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads, dim_feedforward=ff_dim, dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layer
        self.out_linear = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 4, output_size)
        )
        
    def forward(self, full_seq, valid_len, pre_len, **kwargs):
        """
        @param full_seq: Input sequence (batch_size, seq_len)
        @param valid_len: Valid lengths (batch_size,)
        @param pre_len: Prediction length
        """
        self.pre_len = pre_len
        embedded_input = self.embed_layer(full_seq, downstream=True, pre_len=pre_len, **kwargs)  # (batch_size, seq_len, hidden_size)
        src_key_padding_mask = self._generate_padding_mask(full_seq, valid_len)
        # causal_mask = self._generate_causal_mask(full_seq.shape[1], pre_len, embedded_input.device)
        memory = self.encoder(embedded_input, src_key_padding_mask=src_key_padding_mask)
        out = self.out_linear(memory[:,:-pre_len,:])  # (batch_size, seq_len, output_size)
        return out[:, -pre_len:, :]  # (batch_size, pre_len, output_size)
    
    def _generate_padding_mask(self, seq, valid_len):
        """ Creates a mask for padding positions based on valid lengths """
        batch_size, seq_len = seq.shape
        mask = torch.arange(seq_len, device=seq.device).expand(batch_size, seq_len) >= valid_len.unsqueeze(1)
        return mask  # Shape: (batch_size, seq_len)

    def _generate_causal_mask(self, seq_len, pre_len, device):
        """
        Creates a causal mask to prevent attending to future positions for pre_len prediction.
        """
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)  # Upper triangular matrix
        # Mask the future positions after valid_len - pre_len
        mask[seq_len - pre_len:] = float('-inf')
        return mask  # Shape: (seq_len, seq_len)


class ErppLocPredictor(nn.Module, ABC):
    def __init__(self, embed_layer, input_size, lstm_hidden_size, fc_hidden_size, output_size, num_layers, seq2seq=True):
        super().__init__()
        self.__dict__.update(locals())

        self.encoder = nn.LSTM(input_size+1, lstm_hidden_size, num_layers, dropout=0.1, batch_first=True)
        self.decoder = nn.LSTM(input_size+1, lstm_hidden_size, num_layers, dropout=0.1, batch_first=True)
        self.mlp = nn.Sequential(nn.Tanh(), nn.Linear(lstm_hidden_size, fc_hidden_size), nn.Tanh())
        self.event_linear = nn.Linear(fc_hidden_size, output_size)
        self.sos = nn.Parameter(torch.zeros(input_size+1).float(), requires_grad=True)

        self.apply(weight_init)

        self.embed_layer = embed_layer
        self.add_module('embed_layer', self.embed_layer)

    def forward(self, full_seq, valid_len, pre_len, **kwargs):
        """
        :param full_seq: combined historical and target sequence, shape (batch_size, seq_len).
            One row: [l_1, l_2, ..., l_h, 0, 0, 0, l_h+1, ..., l_h+n], where h is the valid length of history sequence,
            n is the length of prediction sequence.
        :param valid_len:
        :param pre_len:
        :param kwargs:
        :return:
        """
        event_embedding = self.embed_layer(full_seq, downstream=True, pre_len=pre_len, **kwargs)
        timestamp = kwargs['timestamp']
        hours = timestamp % (24 * 60 * 60) / 60 / 60 / 24  # (batch, seq_len)

        lstm_input = torch.cat([event_embedding, hours.unsqueeze(-1)], dim=-1)

        if self.seq2seq:
            lstm_out_pre = seq2seq_forward(self.encoder, self.decoder, lstm_input, valid_len, pre_len)
        else:
            lstm_out_pre = rnn_forward(self.encoder, self.sos, lstm_input, valid_len, pre_len)

        mlp_out = self.mlp(lstm_out_pre)
        event_out = self.event_linear(mlp_out)
        return event_out  # (batch_size, pre_len, num_loc)


class StlstmLocPredictor(nn.Module, ABC):
    def __init__(self, embed_layer, num_slots, aux_embed_size, time_thres, dist_thres,
                 input_size, lstm_hidden_size, fc_hidden_size, output_size, num_layers, seq2seq=True):
        super().__init__()
        self.__dict__.update(locals())

        self.time_embed = nn.Embedding(num_slots+1, aux_embed_size)
        self.dist_embed = nn.Embedding(num_slots+1, aux_embed_size)

        self.encoder = nn.LSTM(input_size + 2 * aux_embed_size, lstm_hidden_size, num_layers, dropout=0.1, batch_first=True)
        self.decoder = nn.LSTM(input_size + 2 * aux_embed_size, lstm_hidden_size, num_layers, dropout=0.1, batch_first=True)
        self.out_linear = nn.Sequential(nn.Tanh(), nn.Linear(lstm_hidden_size, fc_hidden_size),
                                        nn.Tanh(), nn.Linear(fc_hidden_size, output_size))
        self.sos = nn.Parameter(torch.zeros(input_size + 2 * aux_embed_size).float(), requires_grad=True)
        self.aux_sos = nn.Parameter(torch.zeros(aux_embed_size * 2).float(), requires_grad=True)
        self.apply(weight_init)

        self.embed_layer = embed_layer
        self.add_module('embed_layer', self.embed_layer)

    def forward(self, full_seq, valid_len, pre_len, **kwargs):
        batch_size = full_seq.size(0)
        # his_len = valid_len - pre_len

        time_delta = kwargs['time_delta'][:, 1:]
        dist = kwargs['dist'][:, 1:]

        time_slot_i = torch.floor(torch.clamp(time_delta, 0, self.time_thres) / self.time_thres * self.num_slots).long()
        dist_slot_i = torch.floor(torch.clamp(dist, 0, self.dist_thres) / self.dist_thres * self.num_slots).long()  # (batch, seq_len-1)
        aux_input = torch.cat([self.aux_sos.reshape(1, 1, -1).repeat(batch_size, 1, 1),
                               torch.cat([self.time_embed(time_slot_i),
                                          self.dist_embed(dist_slot_i)], dim=-1)], dim=1)  # (batch, seq_len, aux_embed_size*2)

        full_embed = self.embed_layer(full_seq, downstream=True, pre_len=pre_len, **kwargs)  # (batch_size, seq_len, input_size)
        lstm_input = torch.cat([full_embed, aux_input], dim=-1)  # (batch_size, seq_len, input_size + aux_embed_size * 2)

        if self.seq2seq:
            lstm_out_pre = seq2seq_forward(self.encoder, self.decoder, lstm_input, valid_len, pre_len)
        else:
            lstm_out_pre = rnn_forward(self.encoder, self.sos, lstm_input, valid_len, pre_len)

        out = self.out_linear(lstm_out_pre)
        return out


class STRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, num_slots, inter_size):
        super().__init__()
        self.__dict__.update(locals())

        self.time_weights = nn.Parameter(torch.zeros(num_slots+1, input_size, inter_size), requires_grad=True)
        self.dist_weights = nn.Parameter(torch.zeros(num_slots+1, inter_size, hidden_size), requires_grad=True)
        self.hidden_weights = nn.Parameter(torch.zeros(hidden_size, hidden_size), requires_grad=True)
        nn.init.xavier_normal_(self.time_weights.data)
        nn.init.xavier_normal_(self.dist_weights.data)
        nn.init.xavier_normal_(self.hidden_weights.data)

    def forward(self, x_context, time_context, dist_context, context_mask, h):
        """
        :param x_context: input context of this step, shape (batch, context_size, input_size)
        :param time_context: time indices corresponding to the context, shape (batch, context_size)
        :param dist_context: distance indices corresponding to the context, shape (batch, context_size)
        :param context_mask: context mask, value with True is a valid place for context fetching, shape (batch, context_size)
        :param h: hidden state from previous step, shape (batch, hidden_size)
        :return: hidden state of this step, shape (batch, hidden_size)
        """
        time_weight = self.time_weights[time_context, :, :]  # (batch, context_size, input_size, inter_size)
        dist_weight = self.dist_weights[dist_context, :, :]  # (batch, context_size, inter_size, hidden_size)
        x_candidate = torch.matmul(x_context.unsqueeze(-2), torch.matmul(time_weight, dist_weight)).squeeze(-2)  # (batch, context_size, hidden_size)
        x_candidate = x_candidate.masked_fill(context_mask.unsqueeze(-1) == False, 0.0).sum(1)  # (batch, hidden_size)
        h_candidate = torch.matmul(h.unsqueeze(-2), self.hidden_weights).squeeze(1)
        return torch.sigmoid(x_candidate + h_candidate)


class STRNN(nn.Module):
    def __init__(self, input_size, hidden_size, inter_size, num_slots):
        super().__init__()
        self.__dict__.update(locals())

        self.strnn_cell = STRNNCell(input_size, hidden_size, num_slots, inter_size)

    def forward(self, x_contexts, time_contexts, dist_contexts, context_masks):
        """
        :param x_contexts: input contexts of each step, shape (batch, seq_len, input_size)
        :param time_contexts:
        :param dist_contexts:
        :param valid_sizes: shape (batch, seq_len)
        :param h:
        :return: output sequence, shape (batch_size, seq_len, hidden_size) and hidden state of last step, shape (1, hidden_size)
        """
        batch_size = x_contexts.size(0)
        seq_len = x_contexts.size(1)

        hidden_state = torch.zeros(batch_size, self.hidden_size).to(x_contexts.device)
        output = []
        for i in range(seq_len):
            x_content = x_contexts[:, :i+1]  # (batch_size, context_size, input_size)
            time_context = time_contexts[:, i, :i+1]
            dist_context = dist_contexts[:, i, :i+1]
            context_mask = context_masks[:, i, :i+1]  # (batch_size, context_size)
            hidden_state = self.strnn_cell(x_content, time_context, dist_context, context_mask, hidden_state)
            output.append(hidden_state)
        return torch.stack(output, dim=1), hidden_state.unsqueeze(0)


class StrnnLocPredictor(nn.Module):
    def __init__(self, embed_layer, num_slots, time_window, dist_window,
                 input_size, hidden_size, inter_size, output_size):
        super().__init__()
        self.__dict__.update(locals())

        self.encoder = STRNN(input_size, hidden_size, inter_size, num_slots)
        self.decoder = STRNN(input_size, hidden_size, inter_size, num_slots)
        self.out_linear = nn.Sequential(nn.Dropout(0.1), nn.Linear(hidden_size, hidden_size * 4),
                                        nn.Tanh(), nn.Linear(hidden_size * 4, output_size))
        self.apply(weight_init)

        self.embed_layer = embed_layer
        self.add_module('embed_layer', self.embed_layer)

    def forward(self, full_seq, valid_len, pre_len, **kwargs):
        batch_size = full_seq.size(0)
        history_len = valid_len - pre_len
        max_len = history_len.max()

        # Generate input sequence.
        full_embed = self.embed_layer(full_seq, downstream=True, pre_len=pre_len, **kwargs)  # (batch, seq_len, embed_size)
        timestamp = kwargs['timestamp']  # (batch, seq_len)
        lat, lng = kwargs['lat'], kwargs['lng']  # (batch, seq_len)
        cat_input = torch.cat([full_embed, timestamp.unsqueeze(-1),
                               lat.unsqueeze(-1), lng.unsqueeze(-1)], dim=-1)  # (batch, seq_len, input_size + 3)
        # After this process, the gap between input sequences will be filled.
        sequential_input = torch.stack([torch.cat([cat_input[i, :s], cat_input[i, -pre_len:],
                                                   torch.zeros(max_len - s, self.input_size + 3).float().to(full_seq.device)], dim=0)
                                        for i, s in enumerate(history_len)], dim=0)  # (batch, seq_len, input_size + 3)
        seq_len = sequential_input.size(1)

        # Calculate a context mask from a given time window.
        seq_timestamp = sequential_input[:, :, -3]  # (batch, seq_len)
        time_delta = seq_timestamp.unsqueeze(-1) - seq_timestamp.unsqueeze(1)
        context_mask = (time_delta <= self.time_window) * \
                       (time_delta >= 0) * \
                       (valid_len.unsqueeze(-1) > torch.arange(seq_len).to(full_seq.device).unsqueeze(0).repeat(batch_size, 1)).unsqueeze(1)  # (batch, seq_len, seq_len)

        # Calculate distances between locations in the trajectory.
        seq_latlng = sequential_input[:, :, -2:]  # (batch, seq_len, 2)
        # (batch, seq_len, 1, 2) - # (batch, 1, seq_len, 2) -> # (batch, seq_len, seq_len, 2)
        dist = (seq_latlng.unsqueeze(2) - seq_latlng.unsqueeze(1)) ** 2
        dist = torch.sqrt(dist.sum(-1))  # (batch, seq_len, seq_len)

        rnn_out, _ = self.encoder(sequential_input[:, :, :-3],
                                  torch.floor(torch.clamp(time_delta, 0, self.time_window) /
                                              self.time_window * self.num_slots).long(),
                                  torch.floor(torch.clamp(dist, 0, self.dist_window) /
                                              self.dist_window * self.num_slots).long(),
                                  context_mask)
        rnn_out_pre = torch.stack([rnn_out[i, s - pre_len-1:s-1] for i, s in enumerate(valid_len)])  # (batch, pre_len, lstm_hidden_size)
        out = self.out_linear(rnn_out_pre)
        return out


class RnnLocPredictor(nn.Module, ABC):
    def __init__(self, embed_layer, input_size, rnn_hidden_size, fc_hidden_size, output_size, num_layers, seq2seq=True):
        super().__init__()
        self.__dict__.update(locals())

        self.encoder = nn.RNN(input_size, rnn_hidden_size, num_layers, dropout=0.1, batch_first=True)
        self.decoder = nn.RNN(input_size, rnn_hidden_size, num_layers, dropout=0.1, batch_first=True)
        self.out_linear = nn.Sequential(nn.Tanh(), nn.Linear(rnn_hidden_size, fc_hidden_size),
                                        nn.LeakyReLU(), nn.Linear(fc_hidden_size, output_size))
        self.sos = nn.Parameter(torch.zeros(input_size).float(), requires_grad=True)
        self.apply(weight_init)

        self.embed_layer = embed_layer
        self.add_module('embed_layer', self.embed_layer)

    def forward(self, full_seq, valid_len, pre_len, **kwargs):
        full_embed = self.embed_layer(full_seq, downstream=True, pre_len=pre_len, **kwargs)
        if self.seq2seq:
            rnn_out_pre = seq2seq_forward(self.encoder, self.decoder, full_embed, valid_len, pre_len)
        else:
            rnn_out_pre = rnn_forward(self.encoder, self.sos, full_embed, valid_len, pre_len)
        out = self.out_linear(rnn_out_pre)
        return out

def loc_prediction(dataset, pre_model, pre_len, num_epoch, batch_size, device):
    pre_model = pre_model.to(device)
    optimizer = torch.optim.Adam(pre_model.parameters(), lr=1e-4)
    loss_func = nn.CrossEntropyLoss()

    def pre_func(batch):
            def _create_src_trg(origin, fill_value):
                src, trg = create_src_trg(origin, pre_len, fill_value)
                full = np.concatenate([src, trg], axis=-1)
                return torch.from_numpy(full).float().to(device)

            user_index, full_seq, weekday, timestamp, length, time_delta, dist, lat, lng = zip(*batch)
            user_index, length = (torch.tensor(item).long().to(device) for item in (user_index, length))

            src_seq, trg_seq = create_src_trg(full_seq, pre_len, fill_value=dataset.num_loc)
            src_seq, trg_seq = (torch.from_numpy(item).long().to(device) for item in [src_seq, trg_seq])
            imp_seq = torch.full_like(trg_seq,fill_value=dataset.num_loc).to(device)
            full_seq = torch.cat([src_seq, imp_seq], dim=-1)

            full_t = _create_src_trg(timestamp, 0)
            full_time_delta = _create_src_trg(time_delta, 0)
            full_dist = _create_src_trg(dist, 0)
            full_lat = _create_src_trg(lat, 0)
            full_lng = _create_src_trg(lng, 0)

            out = pre_model(full_seq, length, pre_len, user_index=user_index, timestamp=full_t,
                            time_delta=full_time_delta, dist=full_dist, lat=full_lat, lng=full_lng)

            out = out.reshape(-1, pre_model.output_size)
            label = trg_seq.reshape(-1)
            return out, label

    train_set = dataset.gen_sequence(min_len=pre_len+1, select_days=0, include_delta=True)
    test_set = dataset.gen_sequence(min_len=pre_len+1, select_days=2, include_delta=True)

    test_point = int(len(train_set) / batch_size / 2)

    save_folder = os.path.join(f"./results/loc_prediction/{pre_model.pre_model_name}", \
        f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")

    os.makedirs(save_folder, exist_ok=True)
    shutil.copy("./downstream/loc_pre.py",save_folder)
    best_acc = 0.0
    best_model_path = os.path.join(save_folder, "best_model.pth")
    # Log file for training results
    log_file = os.path.join(save_folder, "training_log.txt")

    with open(log_file, "w") as log:  # <--- FIXED: Bind log_file to log!
        for epoch in range(num_epoch):
            progress_bar = tqdm(enumerate(next_batch(shuffle(train_set), batch_size)), total=len(train_set) // batch_size, desc=f"Epoch {epoch+1}/{num_epoch}")
            epoch_losses = []
            epoch_scores = []            
            for i, batch in progress_bar:
                out, label = pre_func(batch)
                loss = loss_func(out, label)
                epoch_losses.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
                if (i+1) % test_point == 0:
                    pres_raw, labels = [], []
                    for test_batch in next_batch(test_set, batch_size * 4):
                        test_out, test_label = pre_func(test_batch)
                        pres_raw.append(test_out.detach().cpu().numpy())
                        labels.append(test_label.detach().cpu().numpy())

                    pres_raw, labels = np.concatenate(pres_raw), np.concatenate(labels)
                    pres = pres_raw.argmax(-1)

                    acc, recall = accuracy_score(labels, pres), recall_score(labels, pres, average='macro')
                    f1_micro, f1_macro = f1_score(labels, pres, average='micro'), f1_score(labels, pres, average='macro')
                    epoch_scores.append([acc, recall, f1_micro, f1_macro])

                    progress_bar.set_postfix({'Acc': f"{acc:.4f}", 'Recall': f"{recall:.4f}", 
                                    'F1_micro': f"{f1_micro:.4f}", 'F1_macro': f"{f1_macro:.4f}"})

            # Compute epoch averages
            avg_loss = np.mean(epoch_losses)
            avg_acc, avg_recall, avg_f1_micro, avg_f1_macro = np.mean(epoch_scores, axis=0)
            # Write epoch results to log file
            log.write(f"Epoch {epoch+1} - Loss: {avg_loss:.6f}, Acc: {avg_acc:.6f}, Recall: {avg_recall:.6f}, "
                    f"F1-micro: {avg_f1_micro:.6f}, F1-macro: {avg_f1_macro:.6f}\n")

            # Save best model
            if avg_acc > best_acc:
                best_acc = avg_acc
                torch.save(pre_model.state_dict(), best_model_path)
                log.write(f"New best model saved with Acc: {best_acc:.6f}\n")

    best_acc, best_recall, best_f1_micro, best_f1_macro = np.max(epoch_scores, axis=0)
    print('Acc %.6f, Recall %.6f' % (best_acc, best_recall)) 
    print('F1-micro %.6f, F1-macro %.6f' % (best_f1_micro, best_f1_macro))

    text = f"{pre_model.pre_model_name}_{num_epoch}|{best_acc:.6f}|{best_recall:.6f}|{best_f1_micro:.6f}|{best_f1_macro:.6f}"
    with open("table.txt","a") as file:
        print(text,file=file)


class MCLocPredictor:
    def __init__(self, num_loc):
        self.transfer_mat = np.zeros((num_loc, num_loc))

    def fit(self, sequences):
        """
        @param sequences: sequences for training.
        """
        for sequence in sequences:
            for s, e in zip(sequence[:-1], sequence[1:]):
                self.transfer_mat[s, e] += 1

    def predict(self, src_seq, pre_len):
        pre_seq = []
        s = src_seq[-pre_len-1]
        for i in range(pre_len):
            pre = np.argmax(self.transfer_mat[s])
            pre_seq.append(pre)
            s = pre
        return pre_seq


def mc_next_loc_prediction(dataset, pre_model, pre_len):
    train_seq = list(zip(*dataset.gen_sequence(min_len=pre_len+1, select_days=0)))[1]
    test_seq = list(zip(*dataset.gen_sequence(min_len=pre_len+1, select_days=2)))[1]
    pre_model.fit(train_seq)

    pres, labels = [], []
    for test_row in test_seq:
        pre_row = pre_model.predict(test_row, pre_len)
        pres.append(pre_row)
        labels.append(test_row[-pre_len:])
    pres, labels = np.array(pres).reshape(-1), np.array(labels).reshape(-1)
    acc, recall = accuracy_score(labels, pres), recall_score(labels, pres, average='micro')
    precision, f1 = precision_score(labels, pres, average='micro'), f1_score(labels, pres, average='micro')
    print('Acc %.6f, Recall %.6f' % (acc, recall))
    print('Pre %.6f, f1 %.6f' % (precision, f1))


def fourier_locpred(dataset, pre_model, pre_len, num_epoch, batch_size, device):
    pre_model = pre_model.to(device)
    optimizer = torch.optim.Adam(pre_model.parameters(), lr=1e-4)
    loss_func = nn.CrossEntropyLoss()

    def pre_func(batch):
            def _create_src_trg(origin, fill_value):
                src, trg = create_src_trg(origin, pre_len, fill_value)
                full = np.concatenate([src, trg], axis=-1)
                return torch.from_numpy(full).float().to(device)

            user_index, full_seq, weekday, timestamp, length, time_delta, dist, lat, lng = zip(*batch)
            user_index, length = (torch.tensor(item).long().to(device) for item in (user_index, length))
            
            _, rel_trg_seq = create_src_trg(full_seq, pre_len, fill_value=dataset.num_loc)
            rel_trg_seq = (torch.from_numpy(rel_trg_seq).long().to(device))

            src_seq, trg_seq = create_src_trg(lat, pre_len, fill_value=0.0)
            src_seq, trg_lat_seq = (torch.from_numpy(item).to(device) for item in [src_seq, trg_seq])
            imp_seq = torch.full_like(trg_lat_seq,fill_value=0.0).to(device)
            full_lat = torch.cat([src_seq, imp_seq], dim=-1)

            src_seq, trg_seq = create_src_trg(lng, pre_len, fill_value=0.0)
            src_seq, trg_lng_seq = (torch.from_numpy(item).to(device) for item in [src_seq, trg_seq])
            full_lng = torch.cat([src_seq, imp_seq], dim=-1)

            full_seq = torch.stack([full_lat, full_lng],dim=-1).to(torch.float32)
            # trg_seq  = torch.stack([trg_lat_seq,trg_lng_seq],dim=-1)
            full_t = _create_src_trg(timestamp, 0)
            full_time_delta = _create_src_trg(time_delta, 0)
            full_dist = _create_src_trg(dist, 0)
            full_lat = _create_src_trg(lat, 0)
            full_lng = _create_src_trg(lng, 0)

            out = pre_model(full_seq, length, pre_len, user_index=user_index, timestamp=full_t,
                            time_delta=full_time_delta, dist=full_dist, lat=full_lat, lng=full_lng)

            out = out.reshape(-1, pre_model.output_size)
            label = rel_trg_seq.reshape(-1)
            return out, label

    train_set = dataset.gen_sequence(min_len=pre_len+1, select_days=0, include_delta=True)
    test_set = dataset.gen_sequence(min_len=pre_len+1, select_days=2, include_delta=True)

    test_point = int(len(train_set) / batch_size / 2)

    save_folder = os.path.join(f"./results/loc_prediction/{pre_model.pre_model_name}", \
        f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")

    os.makedirs(save_folder, exist_ok=True)
    shutil.copy("./downstream/loc_pre.py",save_folder)
    best_acc = 0.0
    best_model_path = os.path.join(save_folder, "best_model.pth")
    # Log file for training results
    log_file = os.path.join(save_folder, "training_log.txt")

    with open(log_file, "w") as log:  # <--- FIXED: Bind log_file to log!
        for epoch in range(num_epoch):
            progress_bar = tqdm(enumerate(next_batch(shuffle(train_set), batch_size)), total=len(train_set) // batch_size, desc=f"Epoch {epoch+1}/{num_epoch}")
            epoch_losses = []
            epoch_scores = []            
            for i, batch in progress_bar:
                out, label = pre_func(batch)
                loss = loss_func(out, label)
                epoch_losses.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
                if (i+1) % test_point == 0:
                    pres_raw, labels = [], []
                    for test_batch in next_batch(test_set, batch_size * 4):
                        test_out, test_label = pre_func(test_batch)
                        pres_raw.append(test_out.detach().cpu().numpy())
                        labels.append(test_label.detach().cpu().numpy())

                    pres_raw, labels = np.concatenate(pres_raw), np.concatenate(labels)
                    pres = pres_raw.argmax(-1)

                    acc, recall = accuracy_score(labels, pres), recall_score(labels, pres, average='macro')
                    f1_micro, f1_macro = f1_score(labels, pres, average='micro'), f1_score(labels, pres, average='macro')
                    epoch_scores.append([acc, recall, f1_micro, f1_macro])

                    progress_bar.set_postfix({'Acc': f"{acc:.4f}", 'Recall': f"{recall:.4f}", 
                                    'F1_micro': f"{f1_micro:.4f}", 'F1_macro': f"{f1_macro:.4f}"})

            # Compute epoch averages
            avg_loss = np.mean(epoch_losses)
            avg_acc, avg_recall, avg_f1_micro, avg_f1_macro = np.mean(epoch_scores, axis=0)
            # Write epoch results to log file
            log.write(f"Epoch {epoch+1} - Loss: {avg_loss:.6f}, Acc: {avg_acc:.6f}, Recall: {avg_recall:.6f}, "
                    f"F1-micro: {avg_f1_micro:.6f}, F1-macro: {avg_f1_macro:.6f}\n")

            # Save best model
            if avg_acc > best_acc:
                best_acc = avg_acc
                torch.save(pre_model.state_dict(), best_model_path)
                log.write(f"New best model saved with Acc: {best_acc:.6f}\n")

    best_acc, best_recall, best_f1_micro, best_f1_macro = np.max(epoch_scores, axis=0)
    print('Acc %.6f, Recall %.6f' % (best_acc, best_recall)) 
    print('F1-micro %.6f, F1-macro %.6f' % (best_f1_micro, best_f1_macro))

    text = f"{pre_model.pre_model_name}_{num_epoch}|{best_acc:.6f}|{best_recall:.6f}|{best_f1_micro:.6f}|{best_f1_macro:.6f}"
    with open("table.txt","a") as file:
        print(text,file=file)
