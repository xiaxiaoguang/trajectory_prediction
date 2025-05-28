import os
from collections import Counter
from argparse import ArgumentParser
import pandas as pd

from datasets.dataset import Dataset
from module import *
from downstream.loc_pre import *
from downstream.visit_time_pre import *
from downstream.traj_classify import *

from runner import CTLErunner,HIERrunner,FOURIERrunner,TEASERrunner
from embed.static import DownstreamEmbed

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--device', help='name of the acceleration device to use', type=int, default=0)
    parser.add_argument('--init_param', action='store_true')
    parser.add_argument('--embed_name', help='name of the embedding model to use', type=str, default='ctle')
    parser.add_argument('--embed_size', type=int, default=128)
    parser.add_argument('--hidden_size', type=int, default=None)
    parser.add_argument('--embed_epoch', type=int, default=5)
    parser.add_argument('--pre_model_name',type=str,default='mc')
    parser.add_argument('--task_name', help='name of the downstream task', type=str, default='loc_pre')
    parser.add_argument('--task_epoch', type=int, default=2)
    parser.add_argument('--task_batch_size', type=int, default=64)
    parser.add_argument('--pre_len', type=int, default=3)
    parser.add_argument('--dataset', type=str, default='pek')
    args = parser.parse_args()
    
    device = f"cuda:{args.device}"
    embed_size = args.embed_size
    embed_name = args.embed_name
    embed_epoch = args.embed_epoch
    task_name = args.task_name
    task_epoch = args.task_epoch
    pre_len = args.pre_len
    init_param = args.init_param
    task_batch_size = args.task_batch_size
    pre_model_name = args.pre_model_name
    dataset_name = args.dataset
    hidden_size = args.hidden_size if args.hidden_size is not None else 4*args.embed_size
    
    import time
    a = time.time()
    if dataset_name == 'pek':
        dataset_path = os.path.join('datasets', 'pek.h5')
        split_days = [list(range(9, 12)), [12], [13]]
    elif dataset_name == 'taxi':
        dataset_path = os.path.join('datasets', 'taxi.h5')
        split_days = [list(range(2, 5)), [5], [6]]
        
    raw_df = pd.read_hdf(dataset_path, key='data')
    coor_df = pd.read_hdf(dataset_path, key='poi')
    dataset = Dataset(raw_df, coor_df, split_days)
    print(time.time()-a)
    a = time.time()
    test_seq = list(zip(*dataset.gen_sequence(min_len=pre_len+1, select_days=2)))[1]
    max_seq_len = Counter(dataset.df['user_index'].to_list()).most_common(1)[0][1]
    id2coor_df = dataset.df[['loc_index', 'lat', 'lng']].drop_duplicates('loc_index').set_index('loc_index').sort_index()
    print(time.time()-a)

    if embed_name == 'downstream':
        embed_layer = DownstreamEmbed(dataset.num_loc, embed_size).to(device)
    
    if embed_name == 'teaser':

        teaser_runner = TEASERrunner(
            embed_size=embed_size,
            num_loc=dataset.num_loc,
            embed_epoch=embed_epoch,
            dataset_name=dataset_name,
            w2v_window_size=1,
            skipgram_neg=5
        )
        embed_layer = teaser_runner(dataset, device=device)


    if embed_name == 'fourier':

        fourierrunner = FOURIERrunner(embed_size=embed_size,
                                    num_loc=dataset.num_loc,
                                    embed_epoch=embed_epoch,
                                    dataset_name=dataset_name)
        
        embed_layer = fourierrunner(dataset, device=device)

    if embed_name == 'hier':

        hiertrainer = HIERrunner(embed_size,
                                dataset.num_loc,
                                hidden_size,
                                embed_epoch,
                                max_seq_len,
                                dataset_name=dataset_name)
        embed_layer = hiertrainer(dataset, device=device)
        
    if embed_name == 'ctle':

        ctletrainer = CTLErunner(embed_size,
                   dataset.num_loc,
                   hidden_size,
                   embed_epoch,
                   max_seq_len,
                   init_param=init_param,
                   dataset_name=dataset_name)
        embed_layer = ctletrainer(dataset,device=device)

    if task_name == 'loc_pre':
        pre_model_seq2seq = True

        if pre_model_name == 'mc':
            pre_model = MCLocPredictor(dataset.num_loc)
            mc_next_loc_prediction(dataset, pre_model, pre_len)
        else:
            st_aux_embed_size = 16
            st_num_slots = 10
    
            if pre_model_name == 'erpp':
                pre_model = ErppLocPredictor(embed_layer, input_size=embed_size, lstm_hidden_size=hidden_size,
                                             fc_hidden_size=hidden_size, output_size=dataset.num_loc, num_layers=2, seq2seq=pre_model_seq2seq)
            elif pre_model_name == 'stlstm':
                pre_model = StlstmLocPredictor(embed_layer, num_slots=st_num_slots, aux_embed_size=st_aux_embed_size, time_thres=10800, dist_thres=0.1,
                                               input_size=embed_size, lstm_hidden_size=hidden_size,
                                               fc_hidden_size=hidden_size, output_size=dataset.num_loc, num_layers=2, seq2seq=pre_model_seq2seq)
            elif pre_model_name == 'rnn':
                pre_model = RnnLocPredictor(embed_layer, input_size=embed_size, rnn_hidden_size=hidden_size, fc_hidden_size=hidden_size,
                                            output_size=dataset.num_loc, num_layers=1, seq2seq=pre_model_seq2seq)
            elif pre_model_name == 'gru':
                pre_model = Seq2SeqLocPredictor(embed_layer, input_size=embed_size, hidden_size=hidden_size,
                                                output_size=dataset.num_loc, num_layers=2)
            elif pre_model_name == 'transformer':
                pre_model = TransformerPredictor(embed_layer, input_size=embed_size, hidden_size=hidden_size,
                                                output_size=dataset.num_loc, num_layers=2)
            elif pre_model_name == 'decoder':
                pre_model = DecoderPredictor(embed_layer, input_size=embed_size, hidden_size=hidden_size,
                                                output_size=2, num_layers=2)
            
            pre_model.pre_model_name = pre_model_name
            
            if embed_name == 'fourier':
                fourier_locpred(dataset, pre_model, pre_len=pre_len, num_epoch=task_epoch,
                           batch_size=64, device=device)
            else:
                loc_prediction(dataset, pre_model, pre_len=pre_len, num_epoch=task_epoch,
                           batch_size=64, device=device)
    
    # if task_name == 'time_pre':
    #     pre_model_name = 'erpp'
    #     use_event_loss = True
    
    #     if pre_model_name == 'erpp':
    #         pre_model = ERPPTimePredictor(embed_layer, input_size=embed_size, lstm_hidden_size=hidden_size,
    #                                       hidden_size=hidden_size, output_size=dataset.num_loc, num_layers=2)
    #         erpp_visit_time_prediction(dataset, pre_model, pre_len=pre_len, num_epoch=task_epoch,
    #                                    batch_size=task_batch_size, device=device, use_event_loss=use_event_loss)
    #     if pre_model_name == 'rmtpp':
    #         pre_model = RMTPPTimePredictor(embed_layer, input_size=embed_size, lstm_hidden_size=hidden_size,
    #                                        hidden_size=hidden_size, output_size=dataset.num_loc, num_layers=2)
    #         erpp_visit_time_prediction(dataset, pre_model, pre_len=pre_len, num_epoch=task_epoch,
    #                                    batch_size=task_batch_size, device=device, use_event_loss=use_event_loss)
    #     pre_model = LSTMTimePredictor(embed_layer, input_size=embed_size, lstm_hidden_size=256,
    #                                   fc_hidden_size=256, output_size=dataset.num_loc, num_layers=2)
    #     lstm_visit_time_prediction(dataset, pre_model, num_epoch=task_epoch, batch_size=task_batch_size, device=device)
    
    #     num_time_slots = 48
    #     time_output_type = 'softmax'
    #     output_size = 1 if time_output_type == 'scalar' else num_time_slots
    
    #     pre_model = ScatterVisitTimePredictor(embed_layer, num_time_slots=num_time_slots,
    #                                           input_size=embed_size, lstm_hidden_size=512,
    #                                           fc_hidden_size=256, output_size=output_size, num_layers=2)
    #     scatter_visit_time_prediction(dataset, pre_model, time_output_type=time_output_type,
    #                                   num_epoch=task_epoch, batch_size=task_batch_size, device=device)
    
    # if task_name == 'classify':
    #     traj_pooling_type = 'lstm'
    
    #     pre_model = FCTrajectoryClassifier(pooling_type=traj_pooling_type, input_size=embed_size, hidden_size=hidden_size, output_size=2)
    #     fc_trajectory_classify(dataset, embed_layer, pre_model, num_epoch=task_epoch, batch_size=task_batch_size, device=device)
    