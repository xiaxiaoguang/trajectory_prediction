from embed.hier import *
import os
import torch
import torch.nn as nn
from embed.static import StaticEmbed

class HIERrunner(nn.Module):
    def __init__(self,
                 embed_size,
                 num_loc,
                 hidden_size,
                 embed_epoch,
                 max_seq_len,
                 hier_num_layers=3,
                 hier_week_embed_size=4,
                 hier_hour_embed_size=4,
                 hier_duration_embed_size=4,
                 hier_share=False,
                 batch_size=64,
                 dataset_name='pek',):
        
        super().__init__()
        self.batch_size = batch_size
        self.embed_epoch = embed_epoch

        self.save_folder = "./results/pretraining/hier"
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
            
        self.model_path = f"hier_{hier_num_layers}_{hier_week_embed_size}_{hier_hour_embed_size}_{hier_duration_embed_size}_\
            {embed_size}_{max_seq_len}_{embed_epoch}_{dataset_name}.pth"
        
        self.save_path = os.path.join(self.save_folder, self.model_path)

        self.hier_embedding = HierEmbedding(embed_size, num_loc,
                                            hier_week_embed_size,
                                            hier_hour_embed_size,
                                            hier_duration_embed_size)

        self.hier_model = Hier(self.hier_embedding,
                               hidden_size=hidden_size,
                               num_layers=hier_num_layers,
                               share=hier_share)

    def forward(self, dataset, device='cuda:0'):
        if os.path.exists(self.save_path):
            print("load existing hier embedding")
            embed_layer = torch.load(self.save_path, weights_only=False).to(device)
        else:
            embed_mat = train_hier(dataset,
                                   self.hier_model,
                                   num_epoch=self.embed_epoch,
                                   batch_size=self.batch_size,
                                   device=device)
            embed_layer = StaticEmbed(embed_mat)
            torch.save(embed_layer, self.save_path)
        return embed_layer
