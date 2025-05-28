import os
import torch
import torch.nn as nn
from embed.ctle import *

class CTLErunner(nn.Module):
    def __init__(self,
                embed_size,
                num_loc,
                hidden_size,
                embed_epoch,
                max_seq_len,
                init_param = False,
                ctle_num_layers = 4,
                ctle_num_heads = 8,
                ctle_mask_prop = 0.2,
                batch_size = 64,
                ctle_detach = False,
                encoding_type = 'positional',
                ctle_objective = "mlm",
                dataset_name = 'pek' ,):
        
        super().__init__()
        self.batch_size = batch_size
        self.embed_epoch = embed_epoch
        self.ctle_mask_prop = ctle_mask_prop

        if encoding_type == 'temporal':
            encoding_layer = TemporalEncoding(embed_size)
        else :
            encoding_layer = PositionalEncoding(embed_size, max_seq_len)

        self.save_folder = "./results/pretraining/ctle"
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder) 

        self.model_path = f"ctle_{ctle_num_layers}_{ctle_num_heads}_{encoding_type}_{ctle_mask_prop}_{embed_size}_ \
            {max_seq_len}_{ctle_detach}_{ctle_objective}_{embed_epoch}_{dataset_name}.pth"
        
        self.save_path = os.path.join(self.save_folder,self.model_path)

        obj_models = [MaskedLM(embed_size, num_loc)]
        if ctle_objective == "mh":
            obj_models.append(MaskedHour(embed_size))

        self.obj_models = nn.ModuleList(obj_models)

        ctle_embedding = CTLEEmbedding(encoding_layer, embed_size, num_loc)

        self.ctle_model = CTLE(ctle_embedding, hidden_size, num_layers=ctle_num_layers, num_heads=ctle_num_heads,
                            init_param=init_param, detach=ctle_detach)
        
    def forward(self,dataset,device='cuda:0'):

        if os.path.exists(self.save_path):
            print("load existing ctle embedding")
            embed_layer = torch.load(self.save_path,weights_only=False).to(device)
        else:        
            embed_layer = train_ctle(dataset, self.ctle_model, 
                                     self.obj_models, mask_prop=self.ctle_mask_prop,
                                    num_epoch=self.embed_epoch, 
                                    batch_size=self.batch_size, 
                                    device=device)
            
            torch.save(embed_layer, self.save_path)
        return embed_layer
