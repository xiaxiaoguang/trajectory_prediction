import torch
from torch import nn
import numpy as np
import pickle
import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 100):
        """
        Positional encoding for latitude and longitude.
        :param d_model: Dimension of the output encoding.
        :param max_len: Maximum length of the sequence (not used here but kept for consistency).
        """
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.linear = nn.Sequential(
                nn.Linear(in_features=2, out_features=self.d_model),  
                nn.ReLU(),                                   
                nn.Linear(in_features=self.d_model, out_features=self.d_model),
                nn.Sigmoid()
            )
        self.bilinear = nn.Bilinear(self.d_model, self.d_model, 1)
        self.activation = nn.Sigmoid()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply positional encoding to latitude and longitude.
        :param x: Input tensor of shape [Batch Size, 1, 2].
        :return: Positionally encoded tensor of shape [Batch Size, 1, d_model].
        """
        batch_size = x.shape[0]
        
        # import pdb
        # pdb.set_trace()
        start_out = self.linear(x[...,:2].float())
        dest_out = self.linear(x[...,2:4].float())
        output = self.bilinear(start_out, dest_out)
        output = self.activation(output)
        return dest_out, output
    

class SinEncoding(nn.Module):
    def __init__(self, d_model: int,device = None, max_len: int = 100):
        """
        Positional encoding for latitude and longitude.
        :param d_model: Dimension of the output encoding.
        :param max_len: Maximum length of the sequence (not used here but kept for consistency).
        """
        super(SinEncoding, self).__init__()
        self.d_model = d_model
        self.device = device
        self.linear = nn.Sequential(
                nn.Linear(in_features=self.d_model, out_features=self.d_model),  
                nn.ReLU(),                                   
                nn.Linear(in_features=self.d_model, out_features=self.d_model),
                nn.Sigmoid()   
            )
        self.bilinear = nn.Bilinear(self.d_model, self.d_model, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply positional encoding to latitude and longitude.
        :param x: Input tensor of shape [Batch Size, 1, 2].
        :return: Positionally encoded tensor of shape [Batch Size, 1, d_model].
        """
        batch_size = x.shape[0]

        lat_start = x[...,0].unsqueeze(-1)  # Shape: [B, 1]
        lon_start = x[...,1].unsqueeze(-1)  # Shape: [B, 1]
        
        lat_dest = x[...,2].unsqueeze(-1)  # Shape: [B, 1]
        lon_dest = x[...,3].unsqueeze(-1)  # Shape: [B, 1]
       

        # Generate sinusoidal encodings
        position = torch.arange(self.d_model, dtype=torch.float, device=x.device)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float, device=x.device) * 
                             -(torch.log(torch.tensor(10000.0)) / self.d_model))

        lat_encoding_start = torch.zeros(batch_size, 1, self.d_model, device=x.device)
        lon_encoding_start = torch.zeros(batch_size, 1, self.d_model, device=x.device)

        lat_encoding_start[...,0::2] = torch.sin(lat_start * div_term).unsqueeze(1)  # Even indices
        lat_encoding_start[...,1::2] = torch.cos(lat_start * div_term).unsqueeze(1)  # Odd indices

        lon_encoding_start[...,0::2] = torch.sin(lon_start * div_term).unsqueeze(1)  # Even indices
        lon_encoding_start[...,1::2] = torch.cos(lon_start * div_term).unsqueeze(1)  # Odd indices

        # Combine latitude and longitude encodings
        encoding_start = lat_encoding_start + lon_encoding_start
        lat_encoding_dest = torch.zeros(batch_size, 1, self.d_model, device=x.device)
        lon_encoding_dest = torch.zeros(batch_size, 1, self.d_model, device=x.device)

        lat_encoding_dest[...,0::2] = torch.sin(lat_dest * div_term).unsqueeze(1)  # Even indices
        lat_encoding_dest[...,1::2] = torch.cos(lat_dest * div_term).unsqueeze(1)  # Odd indices

        lon_encoding_dest[...,0::2] = torch.sin(lon_dest * div_term).unsqueeze(1)  # Even indices
        lon_encoding_dest[...,1::2] = torch.cos(lon_dest * div_term).unsqueeze(1)  # Odd indices

        # Combine latitude and longitude encodings
        encoding_dest = lat_encoding_dest + lon_encoding_dest
        
        # output = self.linear(encoding_start)
        output = self.bilinear(encoding_start, encoding_dest)
        output = self.activation(output)
        return encoding_dest, output


class FourierEncoding_IM(nn.Module):
    def __init__(self, d_model: int, device=None, num_frequencies: int = 10):
        super().__init__()

        self.device  = device
        self.d_model = d_model
        self.num_frequencies = num_frequencies
        self.frequencies = torch.linspace(1.0, 100.0, d_model // 2, device=device)
        self.bilinear = nn.Bilinear(self.d_model, self.d_model, 1)
        self.reset_parameters()

    def reset_parameters(self):
        # torch.nn.init.kaiming_uniform_(self.bilinear.weight,a=math.sqrt(5))
        with torch.no_grad():
            self.bilinear.weight /= (self.d_model * self.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        # lat_start = x[...,0] .unsqueeze(-1)  # Shape: [B, 1]
        # lon_start = x[...,1] .unsqueeze(-1)  # Shape: [B, 1]
        # lat_dest = x[...,2].unsqueeze(-1)  # Shape: [B, 1]
        # lon_dest =x[...,3].unsqueeze(-1)  # Shape: [B, 1]
        encoding_start = torch.zeros((batch_size, self.d_model), device=x.device)  # Shape: [B, 1, F*2]
        encoding_dest = torch.zeros_like(encoding_start)  # Shape: [B, 1, F*2]
        frequencies = self.frequencies.repeat(batch_size,1)  # Shape: [F]
        num_freq = frequencies.shape[1]
        encoding_start[..., :num_freq * 2:2] = torch.sin(frequencies.unsqueeze(1) * x[...,:2].unsqueeze(-1)).sum(dim=1)  # Shape: [F, B]
        encoding_start[..., 1:num_freq * 2:2] = torch.cos(frequencies.unsqueeze(1) * x[...,:2].unsqueeze(-1)).sum(dim=1)  # Shape: [F, B]
        encoding_dest[..., :num_freq * 2:2] = torch.sin(frequencies.unsqueeze(1) * x[...,2:].unsqueeze(-1)).sum(dim=1)  # Shape: [F, B] 
        encoding_dest[..., 1:num_freq * 2:2] = torch.cos(frequencies.unsqueeze(1) * x[...,2:].unsqueeze(-1)).sum(dim=1)  # Shape: [F, B]
        # sin_lat_dest = torch.sin(frequencies * lat_dest)  # Shape: [F, B]
        # cos_lat_dest = torch.cos(frequencies * lat_dest)  # Shape: [F, B]
        # sin_lon_dest = torch.sin(frequencies * lon_dest)  # Shape: [F, B]
        # cos_lon_dest = torch.cos(frequencies * lon_dest)  # Shape: [F, B]
        # encoding_start_lat = sin_lat_start + sin_lon_start  # Shape: [F, B]
        # encoding_start_lon = cos_lat_start + cos_lon_start  # Shape: [F, B]
        # encoding_dest_lat = sin_lat_dest + sin_lon_dest  # Shape: [F, B]
        # encoding_dest_lon = cos_lat_dest + cos_lon_dest  # Shape: [F, B]
        # encoding_start[..., :num_freq * 2:2] = encoding_start_lat 
        # encoding_start[..., 1:num_freq * 2:2] = encoding_start_lon 
        # encoding_dest[..., :num_freq * 2:2] = encoding_dest_lat 
        # encoding_dest[..., 1:num_freq * 2:2] = encoding_dest_lon 
        #======================== fusion output ========================
        output = self.bilinear(encoding_start, encoding_dest)

        return encoding_dest,output

class LSTMBasedEncoder(nn.Module):
    def __init__(self, d_model, num_layers=1):
        super(LSTMBasedEncoder, self).__init__()
        self.lstm = nn.LSTM(
            input_size=2,       # Number of input features
            hidden_size=d_model, # Size of the hidden state
            num_layers=num_layers, # Three layers for deeper architecture
            batch_first=True    # Input shape: (batch_size, sequence_length, input_size)
        )
        self.linear1 = nn.Linear(d_model, 1)

    def forward(self, x):
        # Handle different input shapes
        if x.dim() == 1:  # Shape: [5] -> Unsqueeze to [1, 1, 5]
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3 and x.size(1) == 1:  # Shape: [batch_size, 1, 5] -> Squeeze to [batch_size, 5]
            x = x.squeeze(1)
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")

        x_start = x[...,:2]
        x_dest = x[...,2:4]
        # Pass through LSTM
        _, (hidden_start, _) = self.lstm(x_start.float())  # Use first 2 features
        _, (hidden_dest, _) = self.lstm(x_dest.float())  # Use first 2 features
        output = self.linear1(hidden_start[-1].float()+ hidden_dest[-1].float())
        # import pdb
        # pdb.set_trace()
        return _, output  # Return the last hidden state of the last layer

class CNNBasedEncoder(nn.Module):
    def __init__(self, d_model):
        super(CNNBasedEncoder, self).__init__()
        self.conv1 = nn.Conv1d(2, d_model, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(d_model, 1, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        if x.dim() == 1:  # Shape: [5] -> Unsqueeze to [1, 1, 5]
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3 and x.size(1) == 1:  # Shape: [batch_size, 1, 5] -> Squeeze to [batch_size, 5]
            x = x.squeeze(1)
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")

        # Reshape for Conv1d: [batch, channels, seq_len]
        x_start = x[...,:2].permute(0, 2, 1)
        x_dest = x[...,2:4].permute(0, 2, 1)

        # Pass through CNN
        x_start = self.conv1(x_start.float())
        x_start = self.conv2(x_start)
        x_start = self.pool(x_start).squeeze(-1)  # Global max pooling
        
        x_dest = self.conv1(x_dest.float())
        x_dest = self.conv2(x_dest)
        x_dest = self.pool(x_dest).squeeze(-1)  # Global max pooling
        # import pdb
        # pdb.set_trace()
        x = x_start + x_dest
        return x, x

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, device=None, num_heads=4, num_layers=1):
        super(TransformerEncoder, self).__init__()
        self.device=device
        self.embedding = nn.Linear(2, d_model)
        self.embedding2 = nn.Linear(2, d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model, num_heads, dim_feedforward=d_model)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.linear1 = nn.Linear(d_model, 1)
    def forward(self, x):
        if x.dim() == 1:  # Shape: [5] -> Unsqueeze to [1, 1, 5]
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3 and x.size(1) == 1:  # Shape: [batch_size, 1, 5] -> Squeeze to [batch_size, 5]
            x = x.squeeze(1)
        elif x.dim()==2:
            x = x.unsqueeze(1)
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")

        # Embed input
        x_start = self.embedding(x[...,:2].float())
        x_dest = self.embedding2(x[...,2:4].float())

        # Reshape for Transformer: [seq_len, batch, hidden_dim]
        # x = x.permute(1, 0, 2)
        x_start = x_start.permute(1, 0, 2)
        # x_dest = x_dest.permute(1, 0, 2)
        x_out = self.transformer_encoder(x_start + x_dest)
        # x_dest = self.transformer_encoder(x_dest)
        # Reshape back: [batch, seq_len, hidden_dim]
        x_out = x_out.permute(1, 0, 2)
        # x_dest = x_dest.permute(1, 0, 2)
        x = self.linear1(x_out[:, -1, :].float())
        return x, x


# NaiveEncoding
# 一言以蔽之，就是学习两个关于起点和终点的基本向量，然后加起来输出。
class PositionalEncodingnew(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncodingnew, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.linear = nn.Linear(d_model,2)
        self.linear1 = nn.Linear(2,1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        if x.dim() == 1:  # Shape: [5] -> Unsqueeze to [1, 1, 5]
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3 and x.size(1) == 1:  # Shape: [batch_size, 1, 5] -> Squeeze to [batch_size, 5]
            x = x.squeeze(1)
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")
        x_start = x[...,:2].float()
        x_dest = x[...,2:4].float()

        x_start_embedding = self.linear(self.pe[:x_start.size(1), :]).unsqueeze(1)
        x_dest_embedding = self.linear(self.pe[:x_dest.size(1), :]).unsqueeze(1)

        x = self.linear1(x_start + x_start_embedding \
                        + x_dest + x_dest_embedding )
        return x, x
    
class FourierMLPEncoding(nn.Module):
    def __init__(self, d_model: int, device = None,num_frequencies: int = 10):
        super().__init__()
        self.d_model = d_model
        self.device = device
        self.num_frequencies = num_frequencies
        self.lin_dim = nn.Linear(in_features=4, out_features=4)
        self.frequencies = torch.linspace(1.0, 100.0, num_frequencies).to(device)
        self.linear = nn.Sequential(
                nn.Linear(in_features=64, out_features=32),  
                nn.ReLU(),                                   
                nn.Linear(in_features=32, out_features=1),
            )
        self.bilinear = nn.Bilinear(self.d_model, self.d_model, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        
        x = self.lin_dim(x[...,:4].float())
        
        lat_start = x[...,0].unsqueeze(-1)  # Shape: [B, 1]
        lon_start =x[...,1].unsqueeze(-1)  # Shape: [B, 1]
        lat_dest = x[...,2].unsqueeze(-1)  # Shape: [B, 1]
        lon_dest =x[...,3].unsqueeze(-1)  # Shape: [B, 1]

        # Initialize encoding with proper dimensions
        encoding_start = torch.zeros(batch_size, 1, self.d_model, device=x.device)
        encoding_dest = torch.zeros(batch_size, 1, self.d_model, device=x.device)

        for i, freq in enumerate(self.frequencies):
            # Calculate indices ensuring we don't overflow
            if 2*i+1 >= self.d_model:
                break

            # Assign with proper squeezing of dimensions
            encoding_start[...,2*i] = torch.sin(freq * lat_start)  #  [32, 1, 1] transfer to [32, 1]
            encoding_start[...,2*i+1] = torch.cos(freq * lat_start) # Shape: [B, 1]
            
            # Add longitude contribution
            encoding_start[...,2*i] += torch.sin(freq * lon_start)
            encoding_start[...,2*i+1] += torch.cos(freq * lon_start)
        
            # Assign with proper squeezing of dimensions
            encoding_dest[...,2*i] = torch.sin(freq * lat_dest)  #  [32, 1, 1] transfer to [32, 1]
            encoding_dest[...,2*i+1] = torch.cos(freq * lat_dest) # Shape: [B, 1]
            
            # Add longitude contribution
            encoding_dest[...,2*i] += torch.sin(freq * lon_dest)
            encoding_dest[...,2*i+1] += torch.cos(freq * lon_dest)
        
        #======================== fusion output ========================
        # output = self.linear(encoding_start)
        output = self.bilinear(encoding_start, encoding_dest)
        output = self.activation(output)
        return encoding_dest,output


class Traj2vec(nn.Module):
    def __init__(self, act='softmax', d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1):
        super(Traj2vec, self).__init__()
        self.f1 = nn.Linear(300, d_model)
        self.f1_trans = nn.Linear(d_model, 2)
        self.transformer_layer1 = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.d_model = d_model
        self.transformer_encoder1 = nn.TransformerEncoder(self.transformer_layer1, num_layers=1)

        # self.transformer_layer2 = nn.TransformerEncoderLayer(
        #     d_model=d_model,
        #     nhead=nhead,
        #     dim_feedforward=dim_feedforward,
        #     dropout=dropout
        # )
        # self.transformer_encoder2 = nn.TransformerEncoder(self.transformer_layer2, num_layers=1)

        self.d2 = nn.Dropout(0.3)
 
        if act == 'softmax':
            self.activation = lambda x: torch.softmax(x, dim=-1)
        elif act == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = torch.relu

    def forward(self, x):
        # import pdb
        # pdb.set_trace()
        x = self.f1(x.view(-1, 300))
        x1 = self.transformer_encoder1(x)
        x1 = self.d2(x1)
        # x2 = self.d2(x2)
        x1 = self.activation(x1)

        # import pdb
        # pdb.set_trace()
        x2 = self.f1_trans(x1)
        return x1, x2

    def encode(self, x):
        out1 = self.fc1(x)
        out2 = self.activation(self.fc2(x))
        out = torch.cat([out1, out2], 1)
        return out











