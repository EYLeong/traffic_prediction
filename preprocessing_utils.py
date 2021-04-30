# external libraries
import os
import sys
import numpy as np
import math
import json
import torch
from copy import deepcopy
from datetime import datetime
from pathlib import Path

def processed(files_dir, process_dir, overwrite=False):
    '''
    Process traffic data in json file and save them in numpy array
    If overwrite=False, do not process the data if the processed files already exist
    -----------------------------
    :param str files_dir: the directory of the raw dataset
           str files_dir: the directory of the processed output
    -----------------------------
    :returns: None
    '''
    Path(process_dir).mkdir(parents=True, exist_ok=True)
    
    # check if files are already processed
    dataset_path = os.path.join(process_dir, "dataset.npy")
    adj_path = os.path.join(process_dir, "adj.npy")
    metadata_path = os.path.join(process_dir, "metadata.json")
    cat2index_path = os.path.join(process_dir, "cat2index.json")
    
    if (not overwrite and
            (os.path.isfile(dataset_path)
            and os.path.isfile(adj_path)
            and os.path.isfile(metadata_path)
            and os.path.isfile(cat2index_path)
            )
       ):
        # do not run the function if both overwrite is false and all processed files already exist
        return
    
    file_paths = get_ordered_file_path(files_dir)
    
    A, metadata, cat2index = get_adjacency(file_paths[0])
    
    X = []
    
    for data_file_path in file_paths:
        print(f"Processing {data_file_path}")
        features = get_features(data_file_path, metadata, cat2index)
        
        X.append(features)
        
    X = np.transpose(X, (1,2,0)) # (num_vertices, num_features, num_timesteps)

    # save both
    np.save(dataset_path, X)
    np.save(adj_path, A)
    
    # save metadata
    with open(metadata_path, 'w') as outfile:
        json.dump(metadata, outfile, sort_keys=True, indent=4)
    
    with open(cat2index_path, 'w') as outfile:
        json.dump(cat2index, outfile, sort_keys=True, indent=4)
    
    print("Done")

def load(process_dir):
    '''
    Load datasets and adjacency matrixs from numpy file. Also calculates the means and stds
    -----------------------------
    :param str process_dir:  the directory of the processed output
    -----------------------------
    :returns: 
        npy: Adjacency matrix
        npy: Feature matrix
        dict: Metadata
        dict: cat2index
        npy: means 
        npy: stds
    '''
    dataset_path = os.path.join(process_dir, "dataset.npy")
    adj_path = os.path.join(process_dir, "adj.npy")
    metadata_path = os.path.join(process_dir, "metadata.json")
    cat2index_path = os.path.join(process_dir, "cat2index.json")

    A = np.load(adj_path)
    X = np.load(dataset_path)
    X = X.astype(np.float32)

    with open(metadata_path) as json_file:
        metadata = json.load(json_file)
        
    with open(cat2index_path) as json_file:
        cat2index = json.load(json_file)
        
    # Normalization using Z-score method
    means = np.mean(X, axis=(0, 2))
    X = X - means.reshape(1, -1, 1)
    stds = np.std(X, axis=(0, 2))
    X = X / stds.reshape(1, -1, 1)

    return A, X, metadata, cat2index, means, stds

def denormalize(X, stds, means, rounding=False):
    """
    rounding = nearest integer
    Returns the denormalize data
    """
    result = X * stds.reshape(1, -1, 1) + means.reshape(1, -1, 1)
    return np.round(result) if rounding else result

def get_normalized_adj(A):
    """
    Returns the degree normalized adjacency matrix.
    """
    A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    return A_wave

def generate_dataset(X, num_timesteps_input, num_timesteps_output):
    """
    Takes node features for the graph and divides them into multiple samples
    along the time-axis by sliding a window of size (num_timesteps_input+
    num_timesteps_output) across it in steps of 1.
    :param X: Node features of shape (num_vertices, num_features,
    num_timesteps)
    :return:
        - Node features divided into multiple samples. Shape is
          (num_samples, num_vertices, num_features, num_timesteps_input).
        - Node targets for the samples. Shape is
          (num_samples, num_vertices, num_features, num_timesteps_output).
    """
    # Generate the beginning index and the ending index of a sample, which
    # contains (num_points_for_training + num_points_for_predicting) points
    indices = [(i, i + (num_timesteps_input + num_timesteps_output)) for i
               in range(X.shape[2] - (
                num_timesteps_input + num_timesteps_output) + 1)]

    # Save samples
    features, target = [], []
    for i, j in indices:
        features.append(
            X[:, :, i: i + num_timesteps_input].transpose(
                (0, 2, 1)))
        target.append(X[:, 0, i + num_timesteps_input: j])

    return torch.from_numpy(np.array(features)), \
           torch.from_numpy(np.array(target))

def get_adjacency(file_path):
    '''
    Generates the Adjacency matrix of the road network, together with other metadata
    -----------------------------
    :param str file_path: the file path of the dataset
    -----------------------------
    :returns:
        npy: Adjacency matrix
        dict: Metadata (which index in the adjacency matrix corresponds to which road)
        dict: Road category to integer for use as feature
    '''
    with open(file_path, 'r') as traffic_data_file:
        traffic_records = json.load(traffic_data_file)
    
    # Get start, end, length, and find all road categories
    traffic_records_formatted = []
    roadcategory_list = []
    for record in traffic_records:
        record = deepcopy(record)
        lat_long_positions = record['Location'].split()
        record['start_pos'] = ' '. join(lat_long_positions[0:2])
        record['end_pos'] = ' '. join(lat_long_positions[2:4])
        record['length'] = link_length(record['start_pos'], record['end_pos'])
        
        if record['RoadCategory'] not in roadcategory_list:
            roadcategory_list.append(record['RoadCategory'])
        
        traffic_records_formatted.append(record)
        
    traffic_records_formatted.sort(key=lambda x: int(x.get('LinkID')))
    roadcategory_list.sort()
    RoadCat2Index = {}
    for i, cat in enumerate(roadcategory_list):
        RoadCat2Index[cat] = i
    
    # Generate Metadata
    nodes_params_dict = {}
    for i, record in enumerate(traffic_records_formatted):
        record = deepcopy(record)
        new_record = {} # Only keep parameters that don't change with time
        new_record["LinkID"] = record["LinkID"]
        new_record["RoadCategory"] = record["RoadCategory"]
        new_record["RoadName"] = record["RoadName"]
        new_record["start_pos"] = record["start_pos"]
        new_record["end_pos"] = record["end_pos"]
        new_record["length"] = record["length"]
        nodes_params_dict[i] = new_record
    
    # Generating a Directed Adjacency matrix
    '''
    Refer to illustrations
    To find a directed adjacency, we need to check each link(node)
    There is a directed adjacency from Node A to Node B if the end_pos of Node A is the start_pos of Node B

    This involves us having to loop through all nodes with: O(n^2) complexity
    (Computation and speed optimisation is not a concern here as this is pre-generated before training)
    '''
    nodes_count = len(nodes_params_dict)
    A = np.zeros((nodes_count,nodes_count), dtype=int)  
    # Finding the directed edges of the nodes
    for i, i_record in nodes_params_dict.items():
        # print(f'=====> Finding edges for Node: {i}, LinkID: {i_record["LinkID"]}')
        for j, j_record in nodes_params_dict.items():
            if i_record['end_pos'] == j_record['start_pos']:
                # print(f'Found a Directed Edge from Node {i} to Node {j}')
                A[i,j] = 1
    return A, nodes_params_dict, RoadCat2Index

def link_length(start_pos, end_pos):
    """
    Calculation of distance between two lat-long geo positions, using Haversine distance
    ------------------------------------
    :param string start_pos: lat & long separated with a space
    :param string end_pos: lat & long separated with a space
    ------------------------------------
    :returns:
        float: total length of the link
    """
    lat1, lon1 = [float(pos) for pos in start_pos.split()]
    lat2, lon2 = [float(pos) for pos in end_pos.split()]
    radius = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) * math.sin(dlat / 2) +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) * math.sin(dlon / 2))
    d = radius * (2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))
    return d


def get_features(file_path, metadata, cat2index):
    '''
    Generates a Feature matrix
    Note: Feature Matrix, X, would contain the output speedband as well. 
    -----------------------------
    :param str file_path: the file path of the dataset
    -----------------------------
    :returns:
        npy: Feature matrix
    '''
        
    X = []
    # Positions of Features
    # 0. SpeedBand
    # 1. RoadCategory
    # 2. Length of Link
    
    with open(file_path, 'r') as traffic_data_file:
        traffic_records = json.load(traffic_data_file)
        
    traffic_records.sort(key=lambda x: int(x.get('LinkID')))
    for i, record in enumerate(traffic_records):
        features = [record['SpeedBand'],cat2index[record['RoadCategory']],metadata[i]['length']]
        X.append(features)
    
    return np.array(X)

# sort the directories by dates
def get_dates(dir_name):
    date_str_format = "%a_%b_%d_%Y"
    my_date = datetime.strptime(dir_name, date_str_format)
    return my_date

# sorted(os.listdir(raw_trunc_dir), key=get_dates)
def get_day_time(file_name):
    date_str_format = "%H_%M_%S"
    my_date = datetime.strptime(file_name, date_str_format)
    return my_date

def processed_get_day_time(file_path):
    head_tail = os.path.split(file_path)
    file_name = head_tail[-1].split(".")[0]
    return get_day_time(file_name)

def get_ordered_file_path(dir_path):
    sorted_dir_name = sorted(os.listdir(dir_path), key=get_dates)
    sorted_dir = [os.path.join(dir_path, d) for d in sorted_dir_name]

    file_path = []
    for d in sorted_dir:
        list_file = [os.path.join(d,f) for f in os.listdir(d)]
        sorted_list_file = sorted(list_file, key=processed_get_day_time)
        file_path.extend(sorted_list_file)
    return file_path
