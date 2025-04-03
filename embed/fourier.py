import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from sklearn.utils import shuffle
from itertools import zip_longest
from utils import next_batch, weight_init

def gen_random_mask(src_valid_lens, src_len, mask_prop):
    """
    @param src_valid_lens: valid length of sequence, shape (batch_size)
    """
    index_list = []
    for batch, l in enumerate(src_valid_lens):
        mask_count = torch.ceil(mask_prop * l).int()
        masked_index = torch.randperm(l)[:mask_count]
        masked_index += src_len * batch
        index_list.append(masked_index)
    return torch.cat(index_list).long().to(src_valid_lens.device)



class FourierEncoding_IM(nn.Module):
    def __init__(self, d_model: int, embed_size : int, device=None, nhead: int = 4, max_lina : int = 100, num_layers : int = 1):
        super().__init__()
        self.device  = device
        self.d_model = d_model
        self.embed_size = embed_size
        # self.num_frequencies = num_frequencies
        self.frequencies = torch.linspace(1.0, max_lina, d_model // 2, device=device)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=nhead, dim_feedforward=4*self.embed_size, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.linear = nn.Linear(self.d_model, self.embed_size)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        batch_size = x.shape[0]
        frequencies = self.frequencies.repeat(batch_size,1).unsqueeze(1).unsqueeze(1)  # Shape: [B ,F]
        encoding = torch.cat([
            torch.sin(frequencies * x.unsqueeze(-1)),  # Shape: [B, P, 2, num_freq]
            torch.cos(frequencies * x.unsqueeze(-1))   # Shape: [B, P, 2, num_freq]
        ], dim=-2)  # Shape: [B, P, 4, num_freq]
        encoding = encoding.reshape(*encoding.shape[:2], 2 ,-1).sum(dim=-2)  # Shape: [B, P, 2 * num_freq]
        encoding = self.transformer_encoder(encoding)       
        output = self.linear(encoding)
        return output
    
    def to(self, device):
        """Custom to() function to properly move buffers and modules to a new device."""
        self.device = device  # Update stored device information
        self.frequencies = self.frequencies.to(device)  # Move buffer tensor
        return super().to(device)  # Move entire model properly


class MaskedLM2(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear = nn.Linear(input_size, 2)
        self.dropout = nn.Dropout(0.1)
        self.loss_func = nn.MSELoss()

    def forward(self, x, **kwargs):
        origin_tokens = kwargs['origin_tokens']
        lm_pre = self.linear(self.dropout(x))  # (batch, seq_len, 2)
        lm_pre = lm_pre.reshape(-1, 2)  # (batch * seq_len, vocab_size)
        return self.loss_func(lm_pre, origin_tokens)



def train_fourier(dataset, model, obj_models, mask_prop, num_epoch, batch_size, device):
    model = model.to(device)
    obj_models = obj_models.to(device)
    user_ids, src_tokens, src_lat, src_lng, src_lens = zip(*dataset.gen_timeseries(select_days=0))
    optimizer = torch.optim.Adam(list(model.parameters()) , lr=1e-4)
    for epoch in range(num_epoch):
        for batch in tqdm(next_batch(shuffle(list(zip(src_tokens, src_lat, src_lng, src_lens))), batch_size=batch_size)):
            src_batch, src_lat_batch, src_lng_batch, src_len_batch = zip(*batch)

            # Here, i don't have model.num_vocab, since i don't use word embedding, i only use fourier embedding
            # src_batch = np.transpose(np.array(list(zip_longest(*src_batch, fillvalue=model.num_vocab))))
            src_lat_batch = np.transpose(np.array(list(zip_longest(*src_lat_batch, fillvalue=0))))
            src_lng_batch = np.transpose(np.array(list(zip_longest(*src_lng_batch, fillvalue=0))))

            # src_batch = torch.tensor(src_batch).long().to(device)
            src_lat_batch = torch.tensor(src_lat_batch).float().to(device)
            src_lng_batch = torch.tensor(src_lng_batch).float().to(device)
            
            # hour_batch = (src_t_batch % (24 * 60 * 60) / 60 / 60).long()

            batch_len, src_len = src_lng_batch.size(0), src_lng_batch.size(1)
            src_valid_len = torch.tensor(src_len_batch).long().to(device)

            # but the masked still important here. I want to use his Masked language modeling loss
            mask_index = gen_random_mask(src_valid_len, src_len, mask_prop=mask_prop)

            src_lat_batch = src_lat_batch.reshape(-1)
            src_lng_batch = src_lng_batch.reshape(-1)

            # hour_batch = hour_batch.reshape(-1)
            origin_lat_tokens = src_lat_batch[mask_index]  # (num_masked)
            origin_lng_tokens = src_lng_batch[mask_index]  # (num_masked)
            src_l_batch = torch.stack([origin_lat_tokens,origin_lng_tokens],dim=-1) # batchsize*len,2

            # origin_hour = hour_batch[mask_index]
            masked_lat_tokens = src_lat_batch.clone()  
            masked_lat_tokens[mask_index] = 0 
            masked_lat_tokens = masked_lat_tokens.view(batch_len, -1) 

            masked_lng_tokens = src_lng_batch.clone()  
            masked_lng_tokens[mask_index] = 0 
            masked_lng_tokens = masked_lng_tokens.view(batch_len, -1) 

            masked_tokens = torch.stack([masked_lat_tokens,masked_lng_tokens],dim=-1)

            out = model(masked_tokens)  # (batch_size, src_len, embed_size)
            masked_out = out.reshape(-1, model.embed_size)[mask_index]  # (num_masked, embed_size)
            
            loss = 0.

            for obj_model in obj_models:
                loss += obj_model(masked_out, origin_tokens=src_l_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model
