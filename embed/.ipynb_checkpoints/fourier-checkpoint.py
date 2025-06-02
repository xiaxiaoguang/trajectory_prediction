import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.utils import shuffle
from itertools import zip_longest
from module.utils import next_batch, weight_init
import matplotlib.pyplot as plt

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
    def __init__(self, d_model: int, embed_size : int, num_vocab : int,
                  device=None, nhead: int = 4, max_lina : int = 100, num_layers : int = 1):
        super().__init__()
        self.device  = device
        self.d_model = d_model
        self.embed_size = embed_size
        self.num_vocab = num_vocab
        self.frequencies = torch.linspace(1.0, max_lina, self.d_model // 2, device=device)
        # self.token_embed = nn.Embedding(num_vocab+2, d_model // 2, padding_idx=num_vocab)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.embed_size, nhead=nhead, dim_feedforward=4*self.embed_size, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.linear = nn.Linear(self.d_model, self.embed_size)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        batch_size, seq_len = x.shape[0], x.shape[1]
        
        # 修正：frequencies 应该是 [d_model//2] 的一维张量
        frequencies = self.frequencies.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, num_freq]
        
        x_expanded = x.unsqueeze(-1)  # [B, P, 1]
        
        encoding = torch.cat([
            torch.sin(frequencies * x_expanded[:,:,0]),  # [B, P, num_freq]
            torch.cos(frequencies * x_expanded[:,:,1])   # [B, P, num_freq]
        ], dim=-1)  # Shape: [B, P, 2*num_freq] = [B, P, d_model]
        
        # print(f"Encoding shape before linear: {encoding.shape}")
        
        temporal_embedding = self.linear(encoding)  # [B, P, embed_size]
        encoding = self.transformer_encoder(temporal_embedding)
        return encoding

    
    def to(self, device):
        """Custom to() function to properly move buffers and modules to a new device."""
        self.device = device  # Update stored device information
        self.frequencies = self.frequencies.to(device)  # Move buffer tensor
        return super().to(device)  # Move entire model properly


class Masked_GC(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear = nn.Linear(input_size, 2)
        self.dropout = nn.Dropout(0.1)
        self.loss_func = nn.MSELoss()
        self.loss_func2 = F.l1_loss

    def forward(self, x, **kwargs):
        origin_tokens = kwargs['origin_latlng_tokens']
        lm_pre = self.linear(self.dropout(x))  # (batch, seq_len, 2)
        lm_pre = lm_pre.reshape(-1, 2)  # (batch * seq_len, vocab_size)
        loss = self.loss_func2(lm_pre, origin_tokens) + self.loss_func(lm_pre, origin_tokens)
        return loss


class Masked_LC(nn.Module):
    def __init__(self, input_size, num_vocab):
        super().__init__()
        self.num_vocab =  num_vocab
        self.linear = nn.Linear(input_size, num_vocab)
        self.act = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(0.1)
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, x, **kwargs):
        origin_tokens = kwargs['origin_loc_tokens']
        lm_pre = self.act(self.linear(self.dropout(x)))  # (batch, seq_len, 2)
        lm_pre = lm_pre.reshape(-1, self.num_vocab)  # (batch * seq_len, vocab_size)
        loss = self.loss_func(lm_pre, origin_tokens)
        return loss


class Predict_TM(nn.Module):
    def __init__(self, embed_size, nhead, num_layers=1):
        super().__init__()
        self.transfer = nn.Linear(embed_size,embed_size)
        decoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=nhead, dim_feedforward=4*embed_size, batch_first=True)
        self.transformer_decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        self.linear = nn.Linear(embed_size, 1)
        self.loss_func = F.l1_loss

    def forward(self, x, **kwargs):
        timestamp = kwargs['origin_timestamp']
        embedding = kwargs['origin_embedding']
        val_len = kwargs['valid_length'] - 1
        B = val_len.shape[0]
        batch_indices = torch.arange(B, device=val_len.device)
        origin_tokens = timestamp[batch_indices, val_len]
        timestamp[batch_indices, val_len] = 0
        decoder_input = self.transfer(timestamp.unsqueeze(-1) + embedding)
        time_embedding = self.transformer_decoder(decoder_input)
        time_token = self.linear(time_embedding).squeeze(-1)
        selected = time_token[batch_indices, val_len]  # shape [B]
        loss = self.loss_func(selected, origin_tokens)
        return loss

def train_fourier(dataset, model, obj_models, mask_prop, num_epoch, batch_size , device, save_path="./best_model.pth"):
    model = model.to(device)
    obj_models = obj_models.to(device)
    optimizer = torch.optim.Adam(list(model.parameters()), lr=3e-4)

    user_ids, src_tokens, src_lat, src_lng, src_lens,src_time = zip(*dataset.gen_timeseries(select_days=0))
    test_user_ids, test_src_tokens, test_src_lat, test_src_lng, test_src_lens,test_src_time = zip(*dataset.gen_timeseries(select_days=2))
    
    loss_history = []
    best_loss = 1e9
    for epoch in range(num_epoch):
        
        model.train()
        epoch_loss = 0

        for batch in tqdm(next_batch(shuffle(list(zip(src_tokens, src_lat, src_lng, src_lens, src_time))), batch_size=batch_size)):
            src_batch, src_lat_batch, src_lng_batch, src_len_batch, src_timestamp = zip(*batch)


            src_batch = torch.tensor(np.array(list(zip_longest(*src_batch, fillvalue=model.num_vocab)))).long().to(device).transpose(0,1)
            src_lat_batch = torch.tensor(np.array(list(zip_longest(*src_lat_batch, fillvalue=0)))).float().to(device).transpose(0,1)
            src_lng_batch = torch.tensor(np.array(list(zip_longest(*src_lng_batch, fillvalue=0)))).float().to(device).transpose(0,1)
            src_timestamp = torch.tensor(np.array(list(zip_longest(*src_timestamp, fillvalue=0)))).float().to(device).transpose(0,1)
            
            # src_batch = np.transpose(np.array(list(zip_longest(*src_batch, fillvalue=model.num_vocab))))
            # src_lat_batch = np.transpose(np.array(list(zip_longest(*src_lat_batch, fillvalue=0))))
            # src_lng_batch = np.transpose(np.array(list(zip_longest(*src_lng_batch, fillvalue=0))))
            # src_batch = torch.tensor(src_batch).long().to(device)
            # src_lat_batch = torch.tensor(src_lat_batch).float().to(device)
            # src_lng_batch = torch.tensor(src_lng_batch).float().to(device)
            
            batch_len, src_len = src_lng_batch.size(0), src_lng_batch.size(1)
            src_valid_len = torch.tensor(src_len_batch).long().to(device)
            mask_index = gen_random_mask(src_valid_len, src_len, mask_prop=mask_prop)

            src_lat_batch = src_lat_batch.reshape(-1)
            src_lng_batch = src_lng_batch.reshape(-1)
            src_batch = src_batch.reshape(-1)

            origin_loc_tokens = src_batch[mask_index]
            origin_lat_tokens = src_lat_batch[mask_index]
            origin_lng_tokens = src_lng_batch[mask_index]

            src_l_batch = torch.stack([origin_lat_tokens, origin_lng_tokens], dim=-1)

            masked_src_tokens = src_batch.clone()
            masked_src_tokens[mask_index] = 0
            masked_src_tokens = masked_src_tokens.view(batch_len, -1)

            masked_lat_tokens = src_lat_batch.clone()
            masked_lat_tokens[mask_index] = 0
            masked_lat_tokens = masked_lat_tokens.view(batch_len, -1)

            masked_lng_tokens = src_lng_batch.clone()
            masked_lng_tokens[mask_index] = 0
            masked_lng_tokens = masked_lng_tokens.view(batch_len, -1)

            masked_tokens = torch.stack([masked_lat_tokens, masked_lng_tokens], dim=-1)
            out = model(masked_tokens, full_loc_seq=masked_src_tokens)
            masked_out = out.reshape(-1, model.embed_size)[mask_index]
            

            loss = 0.
            for obj_model in obj_models:
                loss += obj_model(masked_out, origin_latlng_tokens=src_l_batch,
                                   origin_loc_tokens=origin_loc_tokens,
                                   origin_embedding=out,
                                   origin_timestamp=src_timestamp,
                                   valid_length=src_valid_len)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(src_tokens):.6f}")
        epoch_loss = 0
        model.eval()
        with torch.no_grad():
            test_loss = 0
            for batch in next_batch(shuffle(list(zip(test_src_tokens, test_src_lat, test_src_lng, test_src_lens, test_src_time))), batch_size=batch_size):
                src_batch, src_lat_batch, src_lng_batch, src_len_batch, src_timestamp = zip(*batch)

                src_batch = torch.tensor(np.array(list(zip_longest(*src_batch, fillvalue=model.num_vocab)))).long().to(device).transpose(0,1)
                src_lat_batch = torch.tensor(np.array(list(zip_longest(*src_lat_batch, fillvalue=0)))).float().to(device).transpose(0,1)
                src_lng_batch = torch.tensor(np.array(list(zip_longest(*src_lng_batch, fillvalue=0)))).float().to(device).transpose(0,1)
                src_timestamp = torch.tensor(np.array(list(zip_longest(*src_timestamp, fillvalue=0)))).float().to(device).transpose(0,1)
                
                batch_len, src_len = src_lng_batch.size(0), src_lng_batch.size(1)
                src_valid_len = torch.tensor(src_len_batch).long().to(device)
                mask_index = gen_random_mask(src_valid_len, src_len, mask_prop=mask_prop)

                src_lat_batch = src_lat_batch.reshape(-1)
                src_lng_batch = src_lng_batch.reshape(-1)
                src_batch = src_batch.reshape(-1)

                origin_loc_tokens = src_batch[mask_index]
                origin_lat_tokens = src_lat_batch[mask_index]
                origin_lng_tokens = src_lng_batch[mask_index]

                src_l_batch = torch.stack([origin_lat_tokens, origin_lng_tokens], dim=-1)

                masked_src_tokens = src_batch.clone()
                masked_src_tokens[mask_index] = 0
                masked_src_tokens = masked_src_tokens.view(batch_len, -1)

                masked_lat_tokens = src_lat_batch.clone()
                masked_lat_tokens[mask_index] = 0
                masked_lat_tokens = masked_lat_tokens.view(batch_len, -1)

                masked_lng_tokens = src_lng_batch.clone()
                masked_lng_tokens[mask_index] = 0
                masked_lng_tokens = masked_lng_tokens.view(batch_len, -1)

                masked_tokens = torch.stack([masked_lat_tokens, masked_lng_tokens], dim=-1)
                out = model(masked_tokens, full_loc_seq=masked_src_tokens)
                masked_out = out.reshape(-1, model.embed_size)[mask_index]
                

                loss = 0.
                for obj_model in obj_models:
                    loss += obj_model(masked_out, origin_latlng_tokens=src_l_batch,
                                    origin_loc_tokens=origin_loc_tokens,
                                    origin_embedding=out,
                                    origin_timestamp=src_timestamp,
                                    valid_length=src_valid_len)

                # for obj_model in obj_models:
                #     test_loss += obj_model(test_masked_out, origin_latlng_tokens=test_src_l_batch, origin_loc_tokens=test_origin_loc_tokens)
                epoch_loss += loss.item()
            
            print(f"Test Loss: {epoch_loss / len(test_src_tokens):.6f}")
            loss_history.append(epoch_loss / len(test_src_tokens))
            if epoch_loss < best_loss :
                best_loss = epoch_loss
                torch.save(model,save_path)
                print(f"update best loss to {best_loss}")
        # Visualization of training loss
        plt.plot(range(0, epoch + 1), loss_history, marker='o', linestyle='-')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Test Loss Over Epochs")
        plt.savefig(save_path+'test.pdf')

    return torch.load(save_path,weights_only=False)