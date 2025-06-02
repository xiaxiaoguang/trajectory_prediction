from torch import nn
import torch
from abc import ABC
from module.utils import weight_init, top_n_accuracy
from module.subutils import seq2seq_forward

class DecoderPredictor(nn.Module, ABC):
    def __init__(self, embed_layer, input_size, hidden_size, output_size, num_layers, num_heads=8, ff_dim=512):
        """
        Initializes a Transformer-based predictor that uses only the encoder for feature extraction.

        @param embed_layer: Embedding layer converting input tokens to dense vectors.
        @param input_size: Dimensionality of input sequences.
        @param hidden_size: Hidden dimension used in transformer encoder.
        @param output_size: Output dimensionality per predicted step.
        @param num_layers: Number of transformer encoder layers.
        @param num_heads: Number of attention heads.
        @param ff_dim: Size of the feedforward layers in the transformer.
        """
        super().__init__()
        self.__dict__.update(locals())        
        self.hidden_size = hidden_size

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_size, nhead=num_heads, dim_feedforward=ff_dim,
            dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.out_linear = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, output_size)
        )

        # Store and optionally freeze the embed_layer
        self.embed_layer = embed_layer
        # for param in self.embed_layer.parameters():
        #     param.requires_grad = False
        self.add_module('embed_layer', self.embed_layer)

    def forward(self, full_seq, valid_len, pre_len, **kwargs):
        """
        Forward pass for prediction using encoder output only.

        @param full_seq: Tensor of shape (batch_size, seq_len) representing full input sequence.
        @param valid_len: Tensor (batch_size,) indicating non-padding lengths in each input.
        @param pre_len: Integer number of prediction steps into the future.

        @return: Output prediction tensor of shape (batch_size, pre_len, output_size).
        """
        self.pre_len = pre_len

        embedded_input = self.embed_layer(full_seq, downstream=True, pre_len=pre_len, **kwargs)
        src_key_padding_mask = self._generate_padding_mask(full_seq, valid_len)
        memory = self.encoder(embedded_input, src_key_padding_mask=src_key_padding_mask)

        out = self.out_linear(memory[:, :-pre_len, :])
        return out[:, -pre_len:, :]

    def _generate_padding_mask(self, seq, valid_len):
        """
        Creates a mask for padding positions in the input sequence.

        @param seq: Tensor of shape (batch_size, seq_len) containing tokenized inputs.
        @param valid_len: Tensor indicating valid lengths per sequence.

        @return: Boolean mask of shape (batch_size, seq_len), where True indicates padding.
        """
        batch_size, seq_len = seq.shape[:2]
        mask = torch.arange(seq_len, device=seq.device).expand(batch_size, seq_len) >= valid_len.unsqueeze(1)
        return mask

    def _generate_causal_mask(self, seq_len, pre_len, device):
        """
        Creates a causal attention mask that restricts attention to previous positions,
        while also excluding future steps reserved for prediction.

        @param seq_len: Total sequence length including prediction span.
        @param pre_len: Number of future steps to predict.
        @param device: Device on which the mask should be created.

        @return: Mask tensor of shape (seq_len, seq_len) with `-inf` above diagonal and in prediction span.
        """
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask[seq_len - pre_len:] = float('-inf')
        return mask



# class DecoderPredictor2(nn.Module, ABC):
#     def __init__(self, embed_layer, input_size, hidden_size, output_size, num_layers, num_heads=8, ff_dim=512):
#         super().__init__()
#         self.__dict__.update(locals())        
        
#         # Store parameters
#         self.embed_layer = embed_layer
#         self.hidden_size = hidden_size
        
#         # Transformer Encoder layer
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=hidden_size, nhead=num_heads, dim_feedforward=ff_dim, dropout=0.1, batch_first=True
#         )
#         self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
#         # Output layer
#         self.out_linear = nn.Sequential(
#             nn.Linear(hidden_size, hidden_size * 4),
#             nn.LeakyReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(hidden_size * 4, output_size)
#         )
        
#     def forward(self, full_seq, valid_len, pre_len, **kwargs):
#         """
#         @param full_seq: Input sequence (batch_size, seq_len)
#         @param valid_len: Valid lengths (batch_size,)
#         @param pre_len: Prediction length
#         """
#         self.pre_len = pre_len
#         embedded_input = self.embed_layer(full_seq, downstream=True, pre_len=pre_len, **kwargs)  # (batch_size, seq_len, hidden_size)
#         src_key_padding_mask = self._generate_padding_mask(full_seq, valid_len)
#         # causal_mask = self._generate_causal_mask(full_seq.shape[1], pre_len, embedded_input.device)
#         memory = self.encoder(embedded_input, src_key_padding_mask=src_key_padding_mask)
#         out = self.out_linear(memory[:,:-pre_len,:])  # (batch_size, seq_len, output_size)
#         return out[:, -pre_len:, :]  # (batch_size, pre_len, output_size)
    
#     def _generate_padding_mask(self, seq, valid_len):
#         """ Creates a mask for padding positions based on valid lengths """
#         batch_size, seq_len = seq.shape
#         mask = torch.arange(seq_len, device=seq.device).expand(batch_size, seq_len) >= valid_len.unsqueeze(1)
#         return mask  # Shape: (batch_size, seq_len)

#     def _generate_causal_mask(self, seq_len, pre_len, device):
#         """
#         Creates a causal mask to prevent attending to future positions for pre_len prediction.
#         """
#         mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)  # Upper triangular matrix
#         # Mask the future positions after valid_len - pre_len
#         mask[seq_len - pre_len:] = float('-inf')
#         return mask  # Shape: (seq_len, seq_len)
