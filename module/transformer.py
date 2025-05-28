from torch import nn
import torch
from abc import ABC
from module.utils import weight_init, top_n_accuracy
from module.subutils import seq2seq_forward

class TransformerPredictor(nn.Module, ABC):
    def __init__(self, embed_layer, input_size, hidden_size, output_size, num_layers, num_heads=8, ff_dim=512):
        """
        Initializes a Transformer-based sequence-to-sequence predictor.

        @param embed_layer: Module to embed input tokens to vectors.
        @param input_size: Dimension of input embeddings.
        @param hidden_size: Hidden dimension for transformer layers.
        @param output_size: Size of the prediction output per time step.
        @param num_layers: Number of encoder and decoder layers.
        @param num_heads: Number of attention heads in each transformer layer.
        @param ff_dim: Dimension of the feedforward layer in transformer blocks.
        """
        super().__init__()
        self.__dict__.update(locals())        
        self.pre_len = None  # Will be set in forward
        self.embed_layer = embed_layer

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads, dim_feedforward=ff_dim,
            dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size, nhead=num_heads, dim_feedforward=ff_dim,
            dropout=0.1, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.out_linear = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 4, output_size)
        )

    def forward(self, full_seq, valid_len, pre_len, **kwargs):
        """
        Performs a forward pass for sequence prediction.

        @param full_seq: Tensor of shape (batch_size, seq_len) containing both historical and future positions.
        @param valid_len: 1D tensor indicating the valid length of each sequence in the batch.
        @param pre_len: Scalar indicating how many steps to predict in the future.

        @return: Output predictions of shape (batch_size, pre_len, output_size).
        """
        self.pre_len = pre_len

        embedded_input = self.embed_layer(full_seq, downstream=True, pre_len=pre_len, **kwargs)

        src_key_padding_mask = self._generate_padding_mask(full_seq, valid_len)
        tgt_mask = self._generate_square_subsequent_mask(pre_len)

        memory = self.encoder(embedded_input, src_key_padding_mask=src_key_padding_mask)

        batch_size = full_seq.shape[0]
        decoder_input = torch.zeros((batch_size, pre_len, self.hidden_size), device=full_seq.device)

        decoder_out = self.decoder(
            decoder_input, memory, tgt_mask=tgt_mask, memory_key_padding_mask=src_key_padding_mask
        )

        out = self.out_linear(decoder_out)
        return out

    def _generate_padding_mask(self, seq, valid_len):
        """
        Generates padding masks for the encoder based on valid sequence lengths.

        @param seq: Input sequence tensor of shape (batch_size, seq_len).
        @param valid_len: 1D tensor indicating the true length of each sequence.
        @return: Boolean mask tensor of shape (batch_size, seq_len), where True indicates padding.
        """
        batch_size, seq_len = seq.shape[:2]
        mask = torch.arange(seq_len, device=seq.device).expand(batch_size, seq_len) >= valid_len.unsqueeze(1)
        return mask

    def _generate_square_subsequent_mask(self, size):
        """
        Creates a causal mask for decoder input to prevent attending to future tokens.

        @param size: Integer indicating the target sequence length.
        @return: Float mask tensor of shape (size, size), with `-inf` above diagonal.
        """
        return torch.triu(torch.ones(size, size, device='cpu') * float('-inf'), diagonal=1).to(dtype=torch.float32)

