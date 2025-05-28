from torch import nn
import torch
from abc import ABC
from module.utils import weight_init, top_n_accuracy
from module.subutils import seq2seq_forward,rnn_forward

class STRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, num_slots, inter_size):
        """
        Spatio-Temporal Recurrent Neural Network Cell.

        Args:
            input_size (int): Dimensionality of input features.
            hidden_size (int): Dimensionality of hidden state.
            num_slots (int): Number of discrete time/distance slots.
            inter_size (int): Intermediate dimensionality for tensor multiplications.
        """
        super().__init__()
        self.__dict__.update(locals())

        # Learnable weight tensors for temporal and spatial relationships
        # Shape: (num_slots+1, input_size, inter_size)
        self.time_weights = nn.Parameter(torch.zeros(num_slots + 1, input_size, inter_size), requires_grad=True)
        # Shape: (num_slots+1, inter_size, hidden_size)
        self.dist_weights = nn.Parameter(torch.zeros(num_slots + 1, inter_size, hidden_size), requires_grad=True)
        # Hidden-to-hidden weight matrix, shape: (hidden_size, hidden_size)
        self.hidden_weights = nn.Parameter(torch.zeros(hidden_size, hidden_size), requires_grad=True)

        # Initialize weights with Xavier normal for stable training
        nn.init.xavier_normal_(self.time_weights.data)
        nn.init.xavier_normal_(self.dist_weights.data)
        nn.init.xavier_normal_(self.hidden_weights.data)

    def forward(self, x_context, time_context, dist_context, context_mask, h):
        """
        Forward pass of STRNNCell.

        Args:
            x_context (Tensor): Input context tensor, shape (batch, context_size, input_size)
            time_context (LongTensor): Time slot indices, shape (batch, context_size)
            dist_context (LongTensor): Distance slot indices, shape (batch, context_size)
            context_mask (BoolTensor): Mask indicating valid context entries, shape (batch, context_size)
            h (Tensor): Previous hidden state, shape (batch, hidden_size)

        Returns:
            Tensor: Updated hidden state, shape (batch, hidden_size)
        """
        # Extract corresponding time weights for each time index in context
        # Result shape: (batch, context_size, input_size, inter_size)
        time_weight = self.time_weights[time_context, :, :]

        # Extract corresponding distance weights for each distance index in context
        # Result shape: (batch, context_size, inter_size, hidden_size)
        dist_weight = self.dist_weights[dist_context, :, :]

        # Calculate the interaction between input context and weights:
        # Step 1: Multiply time_weight and dist_weight: (batch, context_size, input_size, hidden_size)
        combined_weight = torch.matmul(time_weight, dist_weight)

        # Step 2: Multiply input features with combined weight:
        # x_context shape (batch, context_size, input_size) -> unsqueeze to (batch, context_size, 1, input_size)
        # matmul with combined_weight -> (batch, context_size, 1, hidden_size)
        x_context_expanded = x_context.unsqueeze(-2)
        x_candidate = torch.matmul(x_context_expanded, combined_weight).squeeze(-2)  # (batch, context_size, hidden_size)

        # Apply mask: zero out invalid contexts before summing
        x_candidate = x_candidate.masked_fill(~context_mask.unsqueeze(-1), 0.0).sum(dim=1)  # (batch, hidden_size)

        # Hidden state transformation: (batch, hidden_size) x (hidden_size, hidden_size) -> (batch, hidden_size)
        h_candidate = torch.matmul(h, self.hidden_weights)

        # Combine candidates and apply sigmoid activation for gating
        return torch.sigmoid(x_candidate + h_candidate)


class STRNN(nn.Module):
    def __init__(self, input_size, hidden_size, inter_size, num_slots):
        super().__init__()
        self.__dict__.update(locals())

        self.strnn_cell = STRNNCell(input_size, hidden_size, num_slots, inter_size)

    def forward(self, x_contexts, time_contexts, dist_contexts, context_masks):
        """
        :param x_contexts: input contexts of each step, shape (batch, seq_len, input_size)
        :param time_contexts: time slot indices for context, shape (batch, seq_len, context_size)
        :param dist_contexts: distance slot indices for context, shape (batch, seq_len, context_size)
        :param context_masks: mask indicating valid context positions, shape (batch, seq_len, context_size)
        :return: 
            - output sequence of hidden states, shape (batch_size, seq_len, hidden_size)
            - hidden state of the last step, shape (1, hidden_size)

        Explanation:
        The model iterates through each time step in the sequence. For each step, it collects the context 
        (all inputs up to the current time), along with their associated time and distance indices and masks.
        It then updates the hidden state using the STRNNCell, which incorporates spatio-temporal relationships.
        The outputs for all time steps are collected and returned, along with the final hidden state.
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
        self.out_linear = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size * 4),
            nn.Tanh(),
            nn.Linear(hidden_size * 4, output_size)
        )
        self.apply(weight_init)

        self.embed_layer = embed_layer
        self.add_module('embed_layer', self.embed_layer)

    def forward(self, full_seq, valid_len, pre_len, **kwargs):
        """
        :param full_seq: full input sequence, shape (batch, seq_len)
        :param valid_len: length of valid (non-padded) input sequence, shape (batch,)
        :param pre_len: length of prediction part (forecast horizon)
        :return: output predictions, shape (batch, pre_len, output_size)

        Explanation:
        The model first embeds the full input sequence, then concatenates it with time and location information.
        It constructs a sequential input where the history and prediction parts are separated and padded if needed.
        A context mask is computed based on a time window to focus on relevant past events.
        Distances between locations in the sequence are calculated to capture spatial relationships.
        These inputs, along with time and distance indices and the context mask, are fed into the STRNN encoder.
        The relevant hidden states corresponding to the prediction horizon are extracted and passed through
        a feed-forward network to generate the final output predictions.
        """
        batch_size = full_seq.size(0)
        history_len = valid_len - pre_len
        max_len = history_len.max()

        # Generate input embeddings and concatenate auxiliary info
        full_embed = self.embed_layer(full_seq, downstream=True, pre_len=pre_len, **kwargs)
        timestamp = kwargs['timestamp']  # (batch, seq_len)
        lat, lng = kwargs['lat'], kwargs['lng']  # (batch, seq_len)
        cat_input = torch.cat([full_embed, timestamp.unsqueeze(-1),
                               lat.unsqueeze(-1), lng.unsqueeze(-1)], dim=-1)  # (batch, seq_len, input_size + 3)

        # Construct sequential input with padding for each sample
        sequential_input = torch.stack([
            torch.cat([
                cat_input[i, :s],                   # history
                cat_input[i, -pre_len:],            # prediction part
                torch.zeros(max_len - s, self.input_size + 3).float().to(full_seq.device)  # padding
            ], dim=0) for i, s in enumerate(history_len)
        ], dim=0)  # (batch, seq_len, input_size + 3)
        seq_len = sequential_input.size(1)

        # Calculate context mask based on time window
        seq_timestamp = sequential_input[:, :, -3]  # (batch, seq_len)
        time_delta = seq_timestamp.unsqueeze(-1) - seq_timestamp.unsqueeze(1)
        context_mask = (time_delta <= self.time_window) * \
                       (time_delta >= 0) * \
                       (valid_len.unsqueeze(-1) > torch.arange(seq_len).to(full_seq.device).unsqueeze(0).repeat(batch_size, 1)).unsqueeze(1)

        # Calculate distances between locations
        seq_latlng = sequential_input[:, :, -2:]  # (batch, seq_len, 2)
        dist = (seq_latlng.unsqueeze(2) - seq_latlng.unsqueeze(1)) ** 2
        dist = torch.sqrt(dist.sum(-1))  # (batch, seq_len, seq_len)

        # Encode with STRNN
        rnn_out, _ = self.encoder(
            sequential_input[:, :, :-3],  # input features excluding time/lat/lng
            torch.floor(torch.clamp(time_delta, 0, self.time_window) / self.time_window * self.num_slots).long(),
            torch.floor(torch.clamp(dist, 0, self.dist_window) / self.dist_window * self.num_slots).long(),
            context_mask
        )

        # Extract the prediction-related hidden states and generate output
        rnn_out_pre = torch.stack([rnn_out[i, s - pre_len - 1 : s - 1] for i, s in enumerate(valid_len)])
        out = self.out_linear(rnn_out_pre)
        return out
