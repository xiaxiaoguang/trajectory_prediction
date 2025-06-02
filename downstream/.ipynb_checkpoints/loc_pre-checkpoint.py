import numpy as np
import torch
from torch import nn
from module import *

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from sklearn.utils import shuffle
from tqdm import tqdm
from module.utils import next_batch, create_src_trg, weight_init, top_n_accuracy
import os 
import shutil
import datetime

def loc_prediction(dataset, pre_model, pre_len, num_epoch, batch_size, device, embed_name='fourier'):

    pre_model = pre_model.to(device)
    optimizer = torch.optim.Adam(pre_model.parameters(), lr=1e-4)
    loss_func = nn.L1Loss()

    def pre_func(batch):
        def _create_src_trg(origin, fill_value):
            src, trg = create_src_trg(origin, pre_len, fill_value)
            full = np.concatenate([src, trg], axis=-1)
            return torch.from_numpy(full).float().to(device)

        user_index, full_seq, weekday, timestamp, length, time_delta, dist, lat, lng = zip(*batch)
        user_index, length = (torch.tensor(item).long().to(device) for item in (user_index, length))

        src_seq, trg_seq = create_src_trg(full_seq, pre_len, fill_value=dataset.num_loc)
        full_src_seq, full_trg_seq = (torch.from_numpy(item).long().to(device) for item in [src_seq, trg_seq])
        imp_seq = torch.full_like(full_trg_seq, fill_value=dataset.num_loc).to(device)
        full_seq = torch.cat([full_src_seq, imp_seq], dim=-1)

        src_lat, trg_lat = create_src_trg(lat, pre_len, fill_value=0.0)
        src_lng, trg_lng = create_src_trg(lng, pre_len, fill_value=0.0)
        src_lat, trg_lat = (torch.from_numpy(t).float().to(device) for t in (src_lat, trg_lat))
        src_lng, trg_lng = (torch.from_numpy(t).float().to(device) for t in (src_lng, trg_lng))
        label = torch.stack([trg_lat, trg_lng], dim=-1).reshape(-1, 2)

        full_t = _create_src_trg(timestamp, 0)
        full_time_delta = _create_src_trg(time_delta, 0)
        full_dist = _create_src_trg(dist, 0)
        full_lat = _create_src_trg(lat, 0)
        full_lng = _create_src_trg(lng, 0)

        out = pre_model(full_seq, length, pre_len, user_index=user_index, 
                        timestamp=full_t, time_delta=full_time_delta, dist=full_dist,
                        lat=full_lat, lng=full_lng)
        out = out.reshape(-1, 2)
        return out, label

    train_set = dataset.gen_sequence(min_len=pre_len+1, select_days=0, include_delta=True)
    test_set = dataset.gen_sequence(min_len=pre_len+1, select_days=2, include_delta=True)
    test_point = int(len(train_set) / batch_size / 2)

    save_folder = os.path.join(f"./results/loc_prediction/{pre_model.pre_model_name}", 
                               f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(save_folder, exist_ok=True)
    shutil.copy("./downstream/loc_pre.py", save_folder)

    best_mae = float('inf')
    best_model_path = os.path.join(save_folder, "best_model.pth")
    log_file = os.path.join(save_folder, "training_log.txt")

    epoch_scores_all = []

    with open(log_file, "w") as log:
        for epoch in range(num_epoch):
            progress_bar = tqdm(enumerate(next_batch(shuffle(train_set), batch_size)),
                                total=len(train_set) // batch_size,
                                desc=f"Epoch {epoch+1}/{num_epoch}")
            epoch_losses = []
            epoch_scores = []

            for i, batch in progress_bar:
                out, label = pre_func(batch)
                loss = loss_func(out, label)
                epoch_losses.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i + 1) % test_point == 0:
                    preds, targets = [], []
                    for test_batch in next_batch(test_set, batch_size * 4):
                        test_out, test_label = pre_func(test_batch)
                        preds.append(test_out.detach().cpu().numpy())
                        targets.append(test_label.detach().cpu().numpy())
                    preds = np.concatenate(preds)
                    targets = np.concatenate(targets)

                    # Compute regression metrics
                    mae = mean_absolute_error(targets, preds)
                    mse = mean_squared_error(targets, preds)
                    rmse = np.sqrt(mse)
                    mape = np.mean(np.abs((targets - preds) / (targets + 1e-8))) * 100

                    epoch_scores.append([mae, mse, rmse, mape])

                    progress_bar.set_postfix({
                        'MAE': f"{mae:.4f}",
                        'MSE': f"{mse:.4f}",
                        'RMSE': f"{rmse:.4f}",
                        'MAPE': f"{mape:.4f}"
                    })

            avg_loss = np.mean(epoch_losses)
            avg_mae, avg_mse, avg_rmse, avg_mape = np.mean(epoch_scores, axis=0)
            epoch_scores_all.append([avg_mae, avg_mse, avg_rmse, avg_mape])

            log.write(f"Epoch {epoch+1} - Loss: {avg_loss:.6f}, MAE: {avg_mae:.6f}, MSE: {avg_mse:.6f}, "
                      f"RMSE: {avg_rmse:.6f}, MAPE: {avg_mape:.6f}\n")

            if avg_mae < best_mae:
                best_mae = avg_mae
                torch.save(pre_model.state_dict(), best_model_path)
                log.write(f"New best model saved with MAE: {best_mae:.6f}\n")

    epoch_scores_all = np.array(epoch_scores_all)
    best_mae, best_mse, best_rmse, best_mape = np.min(epoch_scores_all, axis=0)

    print('MAE %.6f, MSE %.6f' % (best_mae, best_mse))
    print('RMSE %.6f, MAPE %.6f' % (best_rmse, best_mape))

    text = f"{pre_model.pre_model_name}_{embed_name}_{num_epoch}|{best_mae:.6f}|{best_mse:.6f}|{best_rmse:.6f}|{best_mape:.6f}"
    with open("table.txt", "a") as file:
        print(text, file=file)


def fourier_locpred(dataset, pre_model, pre_len, num_epoch, batch_size, device, embed_name='fourier'):
    pre_model = pre_model.to(device)
    optimizer = torch.optim.Adam(pre_model.parameters(), lr=5e-4)
    loss_func = nn.L1Loss()

    def pre_func(batch):
            def _create_src_trg(origin, fill_value):
                src, trg = create_src_trg(origin, pre_len, fill_value)
                full = np.concatenate([src, trg], axis=-1)
                return torch.from_numpy(full).float().to(device)

            user_index, full_seq, weekday, timestamp, length, time_delta, dist, lat, lng = zip(*batch)
            user_index, length = (torch.tensor(item).long().to(device) for item in (user_index, length))

            full_ind_seq, rel_trg_seq = create_src_trg(full_seq, pre_len, fill_value=dataset.num_loc)
            full_ind_seq, rel_trg_seq = (torch.from_numpy(item).long().to(device) for item in [full_ind_seq, rel_trg_seq])
            imp_seq = torch.full_like(rel_trg_seq,fill_value=dataset.num_loc).to(device)
            full_seq = torch.cat([full_ind_seq, imp_seq], dim=-1)

            src_seq, trg_seq = create_src_trg(lat, pre_len, fill_value=0.0)
            src_seq, trg_lat_seq = (torch.from_numpy(item).to(device) for item in [src_seq, trg_seq])
            imp_seq = torch.full_like(trg_lat_seq,fill_value=0.0).to(device)
            full_lat = torch.cat([src_seq, imp_seq], dim=-1)

            src_seq, trg_seq = create_src_trg(lng, pre_len, fill_value=0.0)
            src_seq, trg_lng_seq = (torch.from_numpy(item).to(device) for item in [src_seq, trg_seq])
            full_lng = torch.cat([src_seq, imp_seq], dim=-1)

            full_latlng = torch.stack([full_lat, full_lng],dim=-1).to(torch.float32)

            full_t = _create_src_trg(timestamp, 0)
            full_time_delta = _create_src_trg(time_delta, 0)
            full_dist = _create_src_trg(dist, 0)
            full_lat = _create_src_trg(lat, 0)
            full_lng = _create_src_trg(lng, 0)
            
            out = pre_model(full_latlng, length, pre_len, user_index=user_index, full_loc_seq=full_seq, timestamp=full_t,
                            time_delta=full_time_delta, dist=full_dist, lat=full_lat, lng=full_lng)

            out = out.reshape(-1, pre_model.output_size)
            
            label = torch.stack([trg_lat_seq, trg_lng_seq],dim=-1).to(torch.float32)
        
            label = label.reshape(-1, pre_model.output_size)
        
            return out, label

    train_set = dataset.gen_sequence(min_len=pre_len+1, select_days=0, include_delta=True)
    test_set = dataset.gen_sequence(min_len=pre_len+1, select_days=2, include_delta=True)

    test_point = int(len(train_set) / batch_size / 2)

    save_folder = os.path.join(f"./results/loc_prediction/{pre_model.pre_model_name}", \
        f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")

    os.makedirs(save_folder, exist_ok=True)
    shutil.copy("./downstream/loc_pre.py",save_folder)
    shutil.copy("./run.py",save_folder)

    best_mae = 0.0
    best_model_path = os.path.join(save_folder, "best_model.pth")
    # Log file for training results
    log_file = os.path.join(save_folder, "training_log.txt")

    with open(log_file, "w") as log:  # <--- FIXED: Bind log_file to log!
        for epoch in range(num_epoch):
            progress_bar = tqdm(enumerate(next_batch(shuffle(train_set), batch_size)), total=len(train_set) // batch_size, desc=f"Epoch {epoch+1}/{num_epoch}")
            epoch_losses = []
            epoch_scores = []            
            for i, batch in progress_bar:
                out, label = pre_func(batch)

                loss = loss_func(out, label)
                epoch_losses.append(loss.item())
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
                if (i+1) % test_point == 0:
                    pres_raw, labels = [], []
                    for test_batch in next_batch(test_set, batch_size * 4):
                        test_out, test_label = pre_func(test_batch)
                        pres_raw.append(test_out.detach().cpu().numpy())
                        labels.append(test_label.detach().cpu().numpy())

                    pres_raw, labels = np.concatenate(pres_raw), np.concatenate(labels)
                    # pres = pres_raw.argmax(-1)
                    pres = pres_raw
    
                    # Compute regression metrics
                    mae = mean_absolute_error(labels, pres)
                    mse = mean_squared_error(labels, pres)
                    rmse = np.sqrt(mse)
                    mape = np.mean(np.abs((np.array(labels) - np.array(pres)) / np.array(labels + 1e-8))) * 100  # Avoid division by zero
                    
                    # Append scores
                    epoch_scores.append([mae, mse, rmse, mape])

                    progress_bar.set_postfix({
                        'MAE': f"{mae:.4f}",
                        'MSE': f"{mse:.4f}",
                        'RMSE': f"{rmse:.4f}",
                        'MAPE': f"{mape:.4f}"
                    })
            avg_loss = np.mean(epoch_losses)
            avg_mae, avg_mse, avg_rmse, avg_mape = np.mean(epoch_scores, axis=0)
            
            log.write(f"Epoch {epoch+1} - Loss: {avg_loss:.6f}, MAE: {avg_mae:.6f}, MSE: {avg_mse:.6f}, "
                      f"RMSE: {avg_rmse:.6f}, MAPE: {avg_mape:.6f}\n")
            
            if avg_mae < best_mae:
                best_mae = avg_mae
                torch.save(pre_model.state_dict(), best_model_path)
                log.write(f"New best model saved with MAE: {best_mae:.6f}\n")
    best_mae, best_mse, best_rmse, best_mape = np.min(epoch_scores, axis=0)
    
    print('MAE %.6f, MSE %.6f' % (best_mae, best_mse))
    print('RMSE %.6f, MAPE %.6f' % (best_rmse, best_mape))
    
    text = f"{pre_model.pre_model_name}_{embed_name}_{num_epoch}|{best_mae:.6f}|{best_mse:.6f}|{best_rmse:.6f}|{best_mape:.6f}"
    with open("table.txt", "a") as file:
        print(text, file=file)

def mc_next_loc_prediction(dataset, pre_model, pre_len):
    train_seq = list(zip(*dataset.gen_sequence(min_len=pre_len+1, select_days=0)))[1]
    test_seq = list(zip(*dataset.gen_sequence(min_len=pre_len+1, select_days=2)))[1]
    pre_model.fit(train_seq)

    pres, labels = [], []
    for test_row in test_seq:
        pre_row = pre_model.predict(test_row, pre_len)
        pres.append(pre_row)
        labels.append(test_row[-pre_len:])
    pres, labels = np.array(pres).reshape(-1), np.array(labels).reshape(-1)
    acc, recall = accuracy_score(labels, pres), recall_score(labels, pres, average='micro')
    precision, f1 = precision_score(labels, pres, average='micro'), f1_score(labels, pres, average='micro')
    print('Acc %.6f, Recall %.6f' % (acc, recall))
    print('Pre %.6f, f1 %.6f' % (precision, f1))



# def loc_prediction(dataset, pre_model, pre_len, num_epoch, batch_size, device):
#     pre_model = pre_model.to(device)
#     optimizer = torch.optim.Adam(pre_model.parameters(), lr=1e-4)
#     loss_func = nn.CrossEntropyLoss()

#     def pre_func(batch):
#             def _create_src_trg(origin, fill_value):
#                 src, trg = create_src_trg(origin, pre_len, fill_value)
#                 full = np.concatenate([src, trg], axis=-1)
#                 return torch.from_numpy(full).float().to(device)

#             user_index, full_seq, weekday, timestamp, length, time_delta, dist, lat, lng = zip(*batch)
#             user_index, length = (torch.tensor(item).long().to(device) for item in (user_index, length))

#             src_seq, trg_seq = create_src_trg(full_seq, pre_len, fill_value=dataset.num_loc)
#             src_seq, trg_seq = (torch.from_numpy(item).long().to(device) for item in [src_seq, trg_seq])
#             imp_seq = torch.full_like(trg_seq,fill_value=dataset.num_loc).to(device)
#             full_seq = torch.cat([src_seq, trg_seq], dim=-1)

#             full_t = _create_src_trg(timestamp, 0)
#             full_time_delta = _create_src_trg(time_delta, 0)
#             full_dist = _create_src_trg(dist, 0)
#             full_lat = _create_src_trg(lat, 0)
#             full_lng = _create_src_trg(lng, 0)

#             out = pre_model(full_seq, length, pre_len, user_index=user_index, timestamp=full_t,
#                             time_delta=full_time_delta, dist=full_dist, lat=full_lat, lng=full_lng)

#             out = out.reshape(-1, pre_model.output_size)
#             label = trg_seq.reshape(-1)
#             return out, label

#     train_set = dataset.gen_sequence(min_len=pre_len+1, select_days=0, include_delta=True)
#     test_set = dataset.gen_sequence(min_len=pre_len+1, select_days=2, include_delta=True)

#     test_point = int(len(train_set) / batch_size / 2)

#     save_folder = os.path.join(f"./results/loc_prediction/{pre_model.pre_model_name}", \
#         f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")

#     os.makedirs(save_folder, exist_ok=True)
#     shutil.copy("./downstream/loc_pre.py",save_folder)
#     best_acc = 0.0
#     best_model_path = os.path.join(save_folder, "best_model.pth")
#     # Log file for training results
#     log_file = os.path.join(save_folder, "training_log.txt")

#     with open(log_file, "w") as log:  # <--- FIXED: Bind log_file to log!
#         for epoch in range(num_epoch):
#             progress_bar = tqdm(enumerate(next_batch(shuffle(train_set), batch_size)), total=len(train_set) // batch_size, desc=f"Epoch {epoch+1}/{num_epoch}")
#             epoch_losses = []
#             epoch_scores = []            
#             for i, batch in progress_bar:
#                 out, label = pre_func(batch)
#                 loss = loss_func(out, label)
#                 epoch_losses.append(loss.item())
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
    
#                 if (i+1) % test_point == 0:
#                     pres_raw, labels = [], []
#                     for test_batch in next_batch(test_set, batch_size * 4):
#                         test_out, test_label = pre_func(test_batch)
#                         pres_raw.append(test_out.detach().cpu().numpy())
#                         labels.append(test_label.detach().cpu().numpy())

#                     pres_raw, labels = np.concatenate(pres_raw), np.concatenate(labels)
#                     pres = pres_raw.argmax(-1)

#                     acc, recall = accuracy_score(labels, pres), recall_score(labels, pres, average='macro')
#                     f1_micro, f1_macro = f1_score(labels, pres, average='micro'), f1_score(labels, pres, average='macro')
#                     epoch_scores.append([acc, recall, f1_micro, f1_macro])

#                     progress_bar.set_postfix({'Acc': f"{acc:.4f}", 'Recall': f"{recall:.4f}", 
#                                     'F1_micro': f"{f1_micro:.4f}", 'F1_macro': f"{f1_macro:.4f}"})

#             avg_loss = np.mean(epoch_losses)
#             avg_acc, avg_recall, avg_f1_micro, avg_f1_macro = np.mean(epoch_scores, axis=0)
#             log.write(f"Epoch {epoch+1} - Loss: {avg_loss:.6f}, Acc: {avg_acc:.6f}, Recall: {avg_recall:.6f}, "
#                     f"F1-micro: {avg_f1_micro:.6f}, F1-macro: {avg_f1_macro:.6f}\n")

#             if avg_acc > best_acc:
#                 best_acc = avg_acc
#                 torch.save(pre_model.state_dict(), best_model_path)
#                 log.write(f"New best model saved with Acc: {best_acc:.6f}\n")

#     best_acc, best_recall, best_f1_micro, best_f1_macro = np.max(epoch_scores, axis=0)
#     print('Acc %.6f, Recall %.6f' % (best_acc, best_recall)) 
#     print('F1-micro %.6f, F1-macro %.6f' % (best_f1_micro, best_f1_macro))

#     text = f"{pre_model.pre_model_name}_{num_epoch}|{best_acc:.6f}|{best_recall:.6f}|{best_f1_micro:.6f}|{best_f1_macro:.6f}"
#     with open("table.txt","a") as file:
#         print(text,file=file)
