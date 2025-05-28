import os
from embed.static import StaticEmbed
from embed.teaser import TeaserData, Teaser, train_teaser
import torch
from torch import nn

class TEASERrunner(nn.Module):
    def __init__(self,
                 embed_size,
                 num_loc,
                 embed_epoch,
                 dataset_name ='pek',
                 w2v_window_size=1,
                 skipgram_neg=5,
                 teaser_num_ne=0,
                 teaser_num_nn=0,
                 teaser_beta=0.0,
                 teaser_week_embed_size=0,
                 teaser_indi_context=False,
                 batch_size=64):
        super().__init__()

        self.embed_size = embed_size
        self.num_loc = num_loc
        self.embed_epoch = embed_epoch
        self.dataset_name = dataset_name
        self.w2v_window_size = w2v_window_size
        self.skipgram_neg = skipgram_neg
        self.teaser_num_ne = teaser_num_ne
        self.teaser_num_nn = teaser_num_nn
        self.teaser_beta = teaser_beta
        self.teaser_week_embed_size = teaser_week_embed_size
        self.teaser_indi_context = teaser_indi_context
        self.batch_size = batch_size

        # File paths
        self.save_folder = "./results/pretraining/teaser"
        os.makedirs(self.save_folder, exist_ok=True)

        self.model_path = f"teaser_{w2v_window_size}_{teaser_num_ne}_{teaser_num_nn}_{teaser_beta}_{teaser_week_embed_size}_{embed_size}_{embed_epoch}.pth"
        self.save_path = os.path.join(self.save_folder, self.model_path)

    def forward(self, dataset, device='cuda:0'):
        if os.path.exists(self.save_path):
            print("load existing teaser embedding")
            embed_layer = torch.load(self.save_path, weights_only=False).to(device)
        else:
            # Prepare sequences
            embed_train_users, embed_train_sentences, embed_train_weekdays, \
            embed_train_timestamp, _length = zip(*dataset.gen_sequence(
                min_len=self.w2v_window_size * 2 + 1,
                select_days=0
            ))

            # Prepare coordinate matrix
            coor_mat = dataset.df[['loc_index', 'lat', 'lng']].drop_duplicates('loc_index').to_numpy()

            # Construct dataset and model
            teaser_dataset = TeaserData(embed_train_users,
                                        embed_train_sentences,
                                        embed_train_weekdays,
                                        coor_mat,
                                        num_ne=self.teaser_num_ne,
                                        num_nn=self.teaser_num_nn,
                                        indi_context=self.teaser_indi_context)

            teaser_model = Teaser(num_vocab=self.num_loc,
                                  num_user=len(dataset.user2index),
                                  embed_dimension=self.embed_size,
                                  week_embed_dimension=self.teaser_week_embed_size,
                                  beta=self.teaser_beta)

            # Train and wrap embedding
            embed_mat = train_teaser(teaser_model,
                                     teaser_dataset,
                                     window_size=self.w2v_window_size,
                                     num_neg=self.skipgram_neg,
                                     batch_size=self.batch_size,
                                     num_epoch=self.embed_epoch,
                                     init_lr=1e-3,
                                     device=device)

            embed_layer = StaticEmbed(embed_mat)
            torch.save(embed_layer, self.save_path)

        return embed_layer
