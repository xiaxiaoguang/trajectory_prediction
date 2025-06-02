import os
import torch
import torch.nn as nn
from embed.fourier import *

class FOURIERrunner(nn.Module):
    def __init__(self,
                 embed_size,
                 num_loc,
                 embed_epoch,
                 dataset_name='pek',
                 d_model=None,
                 max_lina=None,
                 nhead=4,
                 num_layers=2,
                 mask_prob=0.2,
                 batch_size=64):
        super().__init__()

        self.batch_size = batch_size
        self.embed_epoch = embed_epoch
        self.mask_prob = mask_prob

        self.d_model = d_model if d_model is not None else embed_size * 4
        self.max_lina = max_lina if max_lina is not None else embed_size

        self.save_folder = "./results/pretraining/fourier"
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

        self.model_path = f"fourier_{self.d_model}_{self.max_lina}_{nhead}_{num_layers}_{embed_size}_{embed_epoch}_{dataset_name}.pth"
        self.save_path = os.path.join(self.save_folder, self.model_path)

        # Define objectives
        self.obj_models = nn.ModuleList([
            Masked_GC(embed_size)
            # Additional objectives can be added here
        ])

        self.fourier_model = FourierEncoding_IM(
            d_model=self.d_model,
            embed_size=embed_size,
            num_vocab=num_loc,
            nhead=nhead,
            max_lina=self.max_lina,
            num_layers=num_layers,
            device='cpu',
        )

    def forward(self, dataset, device='cuda:0'):
        if os.path.exists(self.save_path):
            print("load existing fourier embedding")
            embed_layer = torch.load(self.save_path, weights_only=False).to(device)
        else:
            # Move model to device
            self.fourier_model.to(device)
            self.obj_models.to(device)

            embed_layer = train_fourier(dataset,
                                        self.fourier_model,
                                        self.obj_models,
                                        mask_prop=self.mask_prob,
                                        num_epoch=self.embed_epoch,
                                        batch_size=self.batch_size,
                                        save_path=self.save_path,
                                        device=device)
            torch.save(embed_layer, self.save_path)
        return embed_layer
