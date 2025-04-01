import yaml
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import time
import datetime
import random
from model.Model_2d import Traj2vec,PositionalEncoding,FourierEncoding_IM, SinEncoding, FourierMLPEncoding, LSTMBasedEncoder,CNNBasedEncoder, TransformerEncoder,PositionalEncodingnew
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import matplotlib.pyplot as plt
import pickle
from torch.utils.data import random_split, DataLoader
import argparse
from tqdm import tqdm
# random.seed(1953)
import math

model_dict = {
    "PositionalEncoding":PositionalEncoding,
    "FourierEncoding_IM" : FourierEncoding_IM,
    "SinEncoding" : SinEncoding,
    "FourierMLPEncoding" : FourierMLPEncoding,
    "LSTMBasedEncoder" : LSTMBasedEncoder,
    "CNNBasedEncoder" : CNNBasedEncoder,
    "TransformerEncoder" : TransformerEncoder,
    "PositionalEncodingnew" : PositionalEncodingnew,
}

def read_graph(dataset):
    """
    Read network edages from text file and return networks object
    :param file: input dataset name
    :return: edage index with shape (n,2)
    """
    dataPath = "./data/" + dataset
    edge = dataPath + "/road/edge_weight.csv"
    node = dataPath + "/road/node.csv"

    df_dege = pd.read_csv(edge, sep=',')
    df_node = pd.read_csv(node, sep=',')

    edge_index = df_dege[["s_node", "e_node"]].to_numpy()
    num_node = df_node["node"].size

    print("{0} road netowrk has {1} edges.".format(config["dataset"], edge_index.shape[0]))
    print("{0} road netowrk has {1} nodes.".format(config["dataset"], num_node))

    return edge_index, num_node

def euclidean_haversine(lat1, lon1, lat2, lon2):
    
    R = 6371000
   
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
   
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    distance = R * c
    distance = distance / 1000
    return distance

def manhattan_haversine(lat1, lon1, lat2, lon2):
    
    R = 6371000
    
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
 
    lat_distance = R * dlat

    lon_distance = R * dlon * math.cos((lat1 + lat2) / 2)
    

    distance = abs(lat_distance) + abs(lon_distance)
    distance = distance / 1000
    return distance
def Euclidean_base_model(loader, device):
    total_loss = 0.0
    rela_mse = 0.0
    total_mae = 0.0
    total_rmse=0.0
    rela_mae = 0.0
    rela_rmse = 0.0
    for data in loader:
        inputs = data[0].to(device).unsqueeze(0)
        # import pdb
        # pdb.set_trace()
        # print("inputs shape:", inputs.shape)
        # euc_dist = torch.cdist(inputs[..., :2], inputs[..., 2:4], p=2)  # Uni: degree
        euc_dist = torch.tensor(euclidean_haversine(inputs[..., 0], inputs[..., 1], inputs[..., 2], inputs[..., 3])).to(device)  # Unit: Kilometers
        
        loss = nn.MSELoss()(euc_dist.float(), inputs[..., 4:5].float())  # Uni: degree
        # loss = nn.MSELoss()(euc_dist, inputs[..., 4:5].float())
        total_loss += loss.item()
        mae = torch.mean(torch.abs(inputs[..., 4:5] - euc_dist.float()))
        total_mae += mae.item()
        mean_squared_true = torch.mean(inputs[..., 4:5])  
        rmse = torch.sqrt(torch.mean((euc_dist.float() - inputs[..., 4:5]) ** 2))
        total_rmse += rmse.item()
        # # Avoid division by zero (add a small epsilon if necessary)
        epsilon = 1e-8
        relative_mse = loss.item() / (mean_squared_true + epsilon)
        # relative_mse = loss.item() / (torch.var(inputs[..., 4:5]) + epsilon)
        rela_mse += relative_mse.item()
        
        
        relative_mae = mae.item() / (mean_squared_true + epsilon)
        rela_mae += relative_mae
        
        relative_rmse = rmse.item() / (mean_squared_true + epsilon)
        rela_rmse += relative_rmse
    avg_train_loss = total_loss / len(loader)
    
    return avg_train_loss, rela_mse / len(loader), total_mae/len(loader), total_rmse/len(loader),rela_mae/len(loader), rela_rmse/len(loader)


def Manhanttan_base_model(loader, device):
    total_loss = 0.0
    rela_mse = 0.0
    total_mae = 0.0
    total_rmse=0.0
    rela_mae = 0.0
    rela_rmse = 0.0
    for data in loader:
        inputs = data[0].to(device).unsqueeze(0)
        # import pdb
        # pdb.set_trace()
        # print("inputs shape:", inputs.shape)
        # euc_dist = torch.cdist(inputs[..., :2], inputs[..., 2:4], p=1)  # Uni: degree
        euc_dist = torch.tensor(manhattan_haversine(inputs[..., 0], inputs[..., 1], inputs[..., 2], inputs[..., 3])).to(device)  # Unit: Kilometers

        loss = nn.MSELoss()(euc_dist.float(), inputs[..., 4:5].float())  # Uni: degree
        # loss = nn.MSELoss()(euc_dist, inputs[..., 4:5].float())
        total_loss += loss.item()
        mae = torch.mean(torch.abs(inputs[..., 4:5] - euc_dist.float()))
        total_mae += mae.item()
        mean_squared_true = torch.mean(inputs[..., 4:5])  
        rmse = torch.sqrt(torch.mean((euc_dist.float() - inputs[..., 4:5]) ** 2))
        total_rmse += rmse.item()
        # # Avoid division by zero (add a small epsilon if necessary)
        epsilon = 1e-8
        relative_mse = loss.item() / (mean_squared_true + epsilon)
        # relative_mse = loss.item() / (torch.var(inputs[..., 4:5]) + epsilon)
        rela_mse += relative_mse.item()
        
        
        relative_mae = mae.item() / (mean_squared_true + epsilon)
        rela_mae += relative_mae
        
        relative_rmse = rmse.item() / (mean_squared_true + epsilon)
        rela_rmse += relative_rmse
    avg_train_loss = total_loss / len(loader)
    
    return avg_train_loss, rela_mse / len(loader), total_mae/len(loader),total_rmse/len(loader),rela_mae/len(loader), rela_rmse/len(loader)


def train(model, loader, optimizer):
    model.train()
    total_loss = 0
    MSE = nn.MSELoss()
    bar = tqdm(loader)
    for data in bar:
        inputs = data.to(model.device)  # Ensure data is a tensor and moved to the correct device
        # last_positions = get_last_positions(inputs) 
        optimizer.zero_grad()
        _ , output = model(inputs)
        loss = MSE(output.float(), inputs[..., 4:5].float())  # Example loss function
        # loss = torch.mean(torch.abs(inputs[..., 4:5] - output))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    avg_train_loss = total_loss / len(loader)
    return avg_train_loss


def evaluate(model, loader):
    """
    Evaluate the model on the validation dataset.
    :param model: The model to evaluate.
    :param loader: DataLoader for the validation dataset.
    :return: Average loss on the validation dataset.
    """
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    rela_mse=0
    # edis = 0
    # mdis = 0
    total_mae = 0
    total_rmse = 0
    rela_mae = 0
    rela_rmse = 0
    with torch.no_grad():  # Disable gradient computation
        for data in loader:
            inputs = data.to(model.device)  # Move data to the correct device
            _ , output = model(inputs)
            
            # output = output * (65.7773 - 0.0046) + 0.0046
            
            loss = nn.MSELoss()(output, inputs[..., 4:5])  # Example loss function
            # loss = torch.mean(torch.abs(inputs[..., 4:5] - output))
            mae = torch.mean(torch.abs(inputs[..., 4:5] - output))
            total_loss += loss.item()
            total_mae += mae.item()
           
            mean_squared_true = torch.mean(inputs[..., 4:5])  

            # # Avoid division by zero (add a small epsilon if necessary)
            rmse = torch.sqrt(torch.mean((output - inputs[..., 4:5]) ** 2))
            total_rmse += rmse.item()
            # # Avoid division by zero (add a small epsilon if necessary)
            epsilon = 1e-8
            relative_mse = loss.item() / (mean_squared_true + epsilon)
            # relative_mse = loss.item() / (torch.var(inputs[..., 4:5]) + epsilon)
            rela_mse += relative_mse.item()
            
            # import pdb
            # pdb.set_trace()
            
            relative_mae = mae.item() / (mean_squared_true + epsilon)
            rela_mae += relative_mae
            
            relative_rmse = rmse.item() / (mean_squared_true + epsilon)
            rela_rmse += relative_rmse
            # print("check val loss and relative loss:", loss.item(), relative_mse.item(), mean_squared_true.item())
            # print("variance of ground truth:", torch.var(inputs[..., 4:5]).item())
    return total_loss / len(loader), rela_mse / len(loader), total_mae/len(loader), total_rmse/len(loader),rela_mae/len(loader), rela_rmse/len(loader)  # Return average loss

def test(model, loader):
    """
    Test the model on the test dataset.
    :param model: The model to test.
    :param loader: DataLoader for the test dataset.
    :return: Average loss on the test dataset.
    """
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    rela_mse = 0
    edis = 0
    mdis = 0
    total_mae = 0
    total_rmse = 0
    rela_mae = 0
    rela_rmse = 0
    with torch.no_grad():  # Disable gradient computation
        for data in loader:
            inputs = data.to(model.device)  # Move data to the correct device
            # last_positions = get_last_positions(inputs)
            
            _ , output = model(inputs)
            # output = output * (65.7773 - 0.0046) + 0.0046
            # import pdb
            # pdb.set_trace()
            # print("inputs[..., ..., 5:6]:", torch.mean(inputs[..., ..., 5:6]), torch.mean(output))
            # output = (output - mean) / (std - mean)
            loss = nn.MSELoss()(output, inputs[..., 4:5])  # Example loss function
            total_loss += loss.item()

            mae = torch.mean(torch.abs(inputs[..., 4:5] - output))
            total_mae += mae.item()
            # Calculate the mean of the squared true values
            # mean_squared_true = torch.mean(last_positions.unsqueeze(1) ** 2)
            mean_squared_true = torch.mean(inputs[..., 4:5])

            rmse = torch.sqrt(torch.mean((output - inputs[..., 4:5]) ** 2))
            total_rmse += rmse.item()
            # # Avoid division by zero (add a small epsilon if necessary)
            epsilon = 1e-8
            relative_mse = loss.item() / (mean_squared_true + epsilon) # useless metrics
            # relative_mse = loss.item() / (torch.var(inputs[..., 4:5]) + epsilon)
            rela_mse += relative_mse.item()
            
            
            relative_mae = mae.item() / (mean_squared_true + epsilon)
            rela_mae += relative_mae
            
            relative_rmse = rmse.item() / (mean_squared_true + epsilon)
            rela_rmse += relative_rmse
            
            # y_mean = torch.mean(inputs[..., 4:5])
            # ss_res = torch.sum((inputs[..., 4:5] - output) ** 2)
            # ss_tot = torch.sum((inputs[..., 4:5] - y_mean) ** 2)
            # r2 = 1 - (ss_res / ss_tot)
            # rela_mse += calculate_r2(inputs[..., 4:5], output)
        
    return total_loss / len(loader), rela_mse / len(loader), total_mae/len(loader), total_rmse/len(loader),rela_mae/len(loader), rela_rmse/len(loader)# Return average loss

def train_with_early_stopping(model, train_loader, val_loader, test_loader, optimizer, patience=5, tol=2e-3):
    """
    Train the model with early stopping based on validation loss and perform final testing.
    :param model: The model to train.
    :param train_loader: DataLoader for the training dataset.
    :param val_loader: DataLoader for the validation dataset.
    :param test_loader: DataLoader for the test dataset.
    :param optimizer: The optimizer for training.
    :param patience: Number of epochs to wait for improvement in validation loss.
    :param tol: Tolerance for early stopping (minimum change in validation loss).
    :return: Training and validation losses, test loss.
    """
    best_val_loss = float('inf')  # Track the best validation loss
    epochs_no_improve = 0  # Track the number of epochs without improvement
    train_losses = []  # Store training losses
    val_losses = []  # Store validation losses

    for epoch in range(200):  # Maximum number of epochs
        # Training
        train_loss = train(model, train_loader, optimizer)
        train_losses.append(train_loss)

        # Validation
        val_loss, re_mse, val_mae, val_rmse, val_rela_mae, val_rela_rmse = evaluate(model, val_loader)
        val_losses.append(val_loss)

        print(f"Epoch: {epoch+1} \tTrain Loss: {train_loss:.4f} \tVal Loss: {val_loss:.4f} \tVal Realtive Error: {re_mse:.4f} \tVal MAE: {val_mae:.4f}\tVal RMSE Error: {val_rmse:.4f} \tVal Relative MAE: {val_rela_mae:.4f}\tVal Relative RMSE: {val_rela_rmse:.4f}")

        # Early stopping check
        if val_loss < best_val_loss - tol:  # Validation loss improved
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:  # Validation loss did not improve
            epochs_no_improve += 1

        if epochs_no_improve >= patience:  # Stop training if no improvement for 'patience' epochs
            print(f"Early stopping at epoch {epoch+1}")
            break

    # np.save("./Training_curve/training_loss_pos.npy", np.array(train_losses))
    # np.save("./Training_curve/val_loss_pos.npy", np.array(val_losses))
    # np.save("./Training_curve/training_loss_sin.npy", np.array(train_losses))
    # np.save("./Training_curve/val_loss_sin.npy", np.array(val_losses))
    # np.save("./Training_curve/training_loss_four_sin.npy", np.array(train_losses))
    # np.save("./Training_curve/val_loss_four_sin.npy", np.array(val_losses))
    # np.save("./Training_curve/training_loss_four_mlp.npy", np.array(train_losses))
    # np.save("./Training_curve/val_loss_four_mlp.npy", np.array(val_losses))
    # Final testing
    test_mse, test_relative_mse, test_mae, test_rmse, test_rela_mae, test_rela_rmse = test(model, test_loader)

    print(f"Test MSE | MAE | RMSE \n {test_mse:.4f}|{test_mae:.4f}|{test_rmse:.4f}")
    return test_mse , test_mae , test_rmse
    # return train_losses, val_losses, test_mse,test_relative_mse
    
def parse_args():
    parser = argparse.ArgumentParser(description="Set model and training parameters.")
    parser.add_argument("--d_model", type=int, default=2048, help="Dimension of the model (default: 64)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate for training (default: 5e-5)")
    parser.add_argument("--repeated",type=int,default=1,help="experiment repeat times")
    parser.add_argument("--model_name",type=str,required=True,help="which model?")
    parser.add_argument("--device",type=int,required=False,default=0,help="which gpu cards?")
    return parser.parse_args()

def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False  # Allow non-deterministic results
        torch.backends.cudnn.benchmark = True       # Improve speed if reproducibility is not a concern

if __name__ == "__main__":
    # config = yaml.safe_load(open('config.yaml'))
    
    # edge_index, num_node = read_graph(str(config["dataset"]))

    # device = "cuda:" + str(config["cuda"])
    # model = Traj2vec(d_model=64).to(device)
    args = parse_args()
    device = "cuda:" + str(args.device)
    #============adopting parameters============
    d_model = args.d_model
    batch_size = args.batch_size
    model = model_dict[args.model_name](d_model, device).to(device)
    # file_path = "./data/tdrive/st_traj/shuffle_coor_list.npy"
    # data = np.load(file_path)
    #================>(loc1, loc2, distance)<================
    # import pdb
    # pdb.set_trace()
    with open('./data/tdrive/position_dis_full_meter.pkl', 'rb') as f:
        loaded_pos = pickle.load(f).to(device)

    print("All data size:", loaded_pos.shape) #torch.Size([2125736, 5])
    
    
    # var = torch.var(loaded_pos[...,-1])
    # mean = torch.mean(loaded_pos[...,-1])
    # std = torch.std(loaded_pos[...,-1])
    # y_norm = (loaded_pos[...,-1] - mean) / std
    # print("Variance of ground truth:", torch.var(y_norm).item())
    # max_value = loaded_pos[...,-1].max()
    # min_value = loaded_pos[...,-1].min()
    # print("Max value of ground truth:", max_value)
    # print("Min value of ground truth:", min_value)
    loaded_pos[...,-1] = loaded_pos[...,-1]/1000
    # max_value = loaded_pos[...,-1].max()
    # min_value = loaded_pos[...,-1].min()
    # loaded_pos[...,-1] = (loaded_pos[...,-1] - min_value) / (max_value - min_value)
    loaded_pos[...,-1] = torch.log(loaded_pos[...,-1])
    max_value = loaded_pos[...,-1].max()
    min_value = loaded_pos[...,-1].min()
    print("Max value of ground truth:", max_value)
    print("Min value of ground truth:", min_value)
    print("Variance of ground truth:", torch.var(loaded_pos[...,-1]).item())

    # target - min / (max - min)
    # println()
    # import pdb
    # pdb.set_trace()
    
    #==== not work ====
    # y_true = loaded_pos[..., 4:5]
    # y_min = torch.min(y_true)
    # y_max = torch.max(y_true)
    # y_norm = (y_true - y_min) / (y_max - y_min)
    # loaded_pos[..., 4:5] = y_norm
    
    
    # y_true = loaded_pos[..., 4:5]
    # mean = torch.mean(y_true)
    # std = torch.std(y_true)
    # y_true_norm = (y_true - mean) / std
    # loaded_pos[..., 4:5] = y_true_norm
    # y_true = loaded_pos[..., 4:5]
    # y_min = torch.min(y_true)
    # y_max = torch.max(y_true)
    # y_true_norm = (y_true - y_min) / (y_max - y_min)
    # loaded_pos[..., 4:5] = y_true_norm
  
    # import pdb
    # pdb.set_trace()
    
    # y_pred_norm = (y_pred - mean) / std
    #==== work ====
    # y_true = loaded_pos[..., 4:5]
    # y_mean = torch.mean(y_true)
    # y_std = torch.std(y_true)
    # y_norm = (y_true - y_mean) / y_std
    # loaded_pos[..., 4:5] = y_norm
    
    #==== not work ====
    # y_true = loaded_pos[..., 4:5]
    # y_max_abs = torch.max(torch.abs(y_true))
    # y_norm = y_true / y_max_abs
    # loaded_pos[..., 4:5] = y_norm
    
    
    
    
    # loaded_pos = loaded_pos.unsqueeze(1)
    # data_train = loaded_pos[:train_size].to(device)
    # data_eval = loaded_pos[train_size:train_size+val_size].to(device)
    # data_test = loaded_pos[train_size+val_size:train_size+val_size+test_size].to(device)
    
    # dataset_train = TensorDataset(data_train)
    # train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

    # dataset_eval = TensorDataset(data_eval)
    # val_loader = DataLoader(dataset_eval, batch_size=batch_size, shuffle=True)

    # dataset_test = TensorDataset(data_test)
    # test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)
    total_length = len(loaded_pos)
    train_size = int(0.6 * total_length)  # 60% training dataset
    val_size = int(0.2 * total_length)    # 20% validation dataset
    test_size = int(0.05 * total_length)   # 5% testing dataset
    # test_size = 20000   # 10% validation dataset
    predict_size = total_length - train_size - val_size - test_size  # the left part is adopted as prediction dataset

    train_data, val_data, test_data, predict_data = random_split(
        loaded_pos, [train_size, val_size, test_size, predict_size]
    )
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    # predict_loader = DataLoader(predict_data, batch_size=batch_size)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    mse = 0
    mae = 0
    rmse = 0
    for i in range(args.repeated):
        results = train_with_early_stopping(
            model, train_loader, val_loader, test_loader, optimizer, patience=15, tol=1e-3
        )
        mse += results[0]
        mae += results[1]
        rmse += results[2]

        set_random_seed(i + 1234)  
        model = model_dict[args.model_name](d_model, device).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    with open("table.out","a") as file: 
        print(f"{args.model_name}|{mse/args.repeated:.4f}|{mae/args.repeated:.4f}|{rmse/args.repeated:.4f}",file=file)
    #========================evaluate========================
    # model.load_state_dict(torch.load("./Model/pos_enc_model.pth"))
    # model.eval()
    # test_mse, test_relative_mse = evaluate(model, test_loader, y_min, y_max)
    # print(f"Test MSE, Relative MSE: {test_mse:.4f},{test_relative_mse:.7f}")
    #========================predict========================
    # model.load_state_dict(torch.load("./Model/pos_enc_model.pth"))
    # model.eval()
    #
    # predict_mse, predict_relative_mse = evaluate(model, predict_loader, y_min, y_max)
    # print(f"Predict MSE, Relative MSE: {predict_mse:.4f},{predict_relative_mse:.7f}")
    #========================test========================
    # model.load_state_dict(torch.load("./Model/pos_enc_model.pth"))
    # model.eval()
    #

    # test_mse, test_relative_mse = test(model, test_loader , y_min, y_max)

    # print(f"Test MSE, Relative MSE: {test_mse:.4f},{test_relative_mse:.7f}")
    
    
    
    # # # #================================baseline comparison based on Euclidean Distance===============================
    # test_mse, test_relative_mse, test_mae,test_rmse, test_rela_mae, test_rela_rmse = Euclidean_base_model(test_loader, device)
    # print(f"Test MSE, Relative Error, MAE, RMSE, Relative MAE, Relative RMSE: {test_mse:.4f},{test_relative_mse:.7f},{test_mae:.4f},{test_rmse:.4f},{test_rela_mae:.4f},{test_rela_rmse:.4f}")
    
    # #================================baseline comparison based on Manhattan Distance===============================
    # test_mse, test_relative_mse, test_mae,test_rmse, test_rela_mae, test_rela_rmse =  Manhanttan_base_model(test_loader, device)
    # print(f"Test MSE, Relative Error, MAE, RMSE, Relative MAE, Relative RMSE: {test_mse:.4f},{test_relative_mse:.7f},{test_mae:.4f},{test_rmse:.4f},{test_rela_mae:.4f},{test_rela_rmse:.4f}")
    
    
    
    
    
    
    
    
    
    
    