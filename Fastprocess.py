import yaml
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pickle
import math

# Haversine function
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
   
    lat1, lon1, lat2, lon2 = torch.deg2rad(lat1), torch.deg2rad(lon1), torch.deg2rad(lat2), torch.deg2rad(lon2)
   
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
    
    distance = R * c
    return distance



def manhattan_distance(output, target):
    # Extract latitude and longitude
    lat1, lon1 = output[..., 0], output[..., 1]
    lat2, lon2 = target[..., 0], target[..., 1]

    # Calculate Manhattan distance
    distance = torch.abs(lat1 - lat2) + torch.abs(lon1 - lon2)

    return distance

def calculate_adjacent_distances(batch):
    """
    Calculate the Euclidean distances between all adjacent points in a batch of trajectories,
    and combine them into [current_points, next_points, distance] format.
    :param batch: Input tensor of shape [Batch Size, L, 2], where the last dimension is (latitude, longitude).
    :return: List of [current_points, next_points, distance].
    """

    # Extract latitude and longitude for current and next points
    lat1, lon1 = batch[:, :-1, 0], batch[:, :-1, 1]
    lat2, lon2 = batch[:, 1:, 0], batch[:, 1:, 1]
    # Compute distances in a vectorized way
    adjacent_distances = haversine(lat1, lon1, lat2, lon2)
    
    return adjacent_distances

def calculate_trajectory_distance_with_gap_unique_filtered(batch):
    """
    Calculate the trajectory distance between a current point and any other point
    that is at least two points ahead in the trajectory, filtering out pairs
    where either point is [0.0, 0.0]. The trajectory distance is defined as
    the sum of adjacent distances between the two points.
    
    :param batch: Tensor of shape [B, L, 2] (latitude, longitude in degrees).
    :return: Tensor with each row as [current_lat, current_lon, next_lat, next_lon, trajectory_distance]
             for valid unique pairs.
    """
    B, L, _ = batch.shape

    # Compute adjacent distances and then the cumulative sum to quickly compute segment distances.
    adjacent_distances = calculate_adjacent_distances(batch)  # shape: [B, L-1]

    # Create cumulative sum along each trajectory; prepend a zero so that cum_adj has shape [B, L]
    cum_adj = torch.cat([torch.zeros(B, 1, device=batch.device), torch.cumsum(adjacent_distances, dim=1)], dim=1)

    # Generate all index pairs (j, k) with k >= j + 2 to skip adjacent points
    idx = torch.triu_indices(L, L, offset=2)
    idx_j, idx_k = idx[0], idx[1]  # Both have shape [P]

    # Gather current points (at index j) and next points (at index k) for all batch elements
    current_points = batch[:, idx_j, :]  # shape: [B, P, 2]
    next_points = batch[:, idx_k, :]     # shape: [B, P, 2]

    # Compute trajectory distances: distance from point j to point k is cum_adj[k] - cum_adj[j]
    traj_distance = cum_adj[:, idx_k] - cum_adj[:, idx_j]  # shape: [B, P]

    # Filter out pairs where either point is [0.0, 0.0]
    valid_mask = (current_points.abs().sum(dim=2) > 0) & (next_points.abs().sum(dim=2) > 0)  # shape: [B, P]

    # Define a function to truncate numbers to a fixed number of decimal places
    def truncate(tensor, decimals=9):
        factor = 10 ** decimals
        return torch.round(tensor * factor) / factor

    # Flatten the valid pairs across batches
    valid_current =truncate(current_points[valid_mask])  # shape: [N, 2]
    valid_next = truncate(next_points[valid_mask])          # shape: [N, 2]
    valid_distance = truncate(traj_distance[valid_mask])      # shape: [N]

    # Combine the results into one tensor with columns: current_lat, current_lon, next_lat, next_lon, trajectory_distance
    result_tensor = torch.stack([
        valid_current[:, 0],
        valid_current[:, 1],
        valid_next[:, 0],
        valid_next[:, 1],
        valid_distance
    ], dim=1)

    # pairs = torch.cat([valid_current, valid_next], dim=1)  # shape: [N, 4]
    # sorted_indices = torch.lexsort([pairs[:, 3], pairs[:, 2], pairs[:, 1], pairs[:, 0]])
    # sorted_pairs = pairs[sorted_indices]
    unique_result_tensor,unique_indices,count = torch.unique(result_tensor, dim=0, sorted=False, return_inverse=True, return_counts=True)
    # unique_result_tensor = result_tensor[unique_indices]
    return unique_result_tensor

if __name__ == "__main__":

    file_path = "../data/tdrive/st_traj/shuffle_coor_list.npy"
    data = torch.tensor(np.load(file_path))

    print("All data size:", data.shape)
    breakpoint()
    results = calculate_trajectory_distance_with_gap_unique_filtered(data)  
    
    print("Preprocessed Tensor size is:", results.size())

    with open('../data/tdrive/position_dis_full_meter.pkl', 'wb') as f:
        pickle.dump(results, f)

    print("Tensor saved to position_dis_full_meter.pkl")



