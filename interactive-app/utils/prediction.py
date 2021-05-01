import streamlit as st
import pandas as pd
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import optim, nn
import os
from os.path import dirname, abspath
import sys
sys.path.append('..')
import preprocessing_utils
import model_utils
import model


@st.cache(persist=True)
def load_data(data):
    return data


@st.cache(persist=True)
def split_data(data):
    return data

#@st.cache(suppress_st_warning=True)
def calculate(num_input, num_output):
    time.sleep(2)
    dd = dirname(dirname(dirname(abspath(__file__))))
    ddd = os.path.join(dd, 'truncated_data')
    print("here",dd, ddd)
    print("syspath", sys.path[0])
    return num_input + num_output

def predict(num_timesteps_input, num_timesteps_output):
    device = torch.device('cpu')
    loss_criterion = nn.MSELoss()
    interactive_app_path = dirname(dirname(abspath(__file__))) # Use this. Having issues with Heroku path system
    raw_dir = os.path.join(interactive_app_path, 'data', 'raw')
    process_dir = os.path.join(interactive_app_path, 'data', 'processed')
    preprocessing_utils.processed(raw_dir, process_dir, overwrite=True)
    A, X, metadata, cat2index, means, stds = preprocessing_utils.load(process_dir)
    test_original_data = X
    test_input, test_target = preprocessing_utils.generate_dataset(test_original_data,
                                                                   num_timesteps_input=num_timesteps_input,
                                                                   num_timesteps_output=num_timesteps_output)
    adj_mat = preprocessing_utils.get_normalized_adj(A)
    adj_mat = torch.from_numpy(adj_mat).to(device)

    i = 0
    indices = [(i, i + (num_timesteps_input + num_timesteps_output))]
    features, target = [], []
    for i, j in indices:
        features.append(
            X[:, :, i: i + num_timesteps_input].transpose(
                (0, 2, 1)))
        target.append(X[:, 0, i + num_timesteps_input: j])

    test_input = torch.from_numpy(np.array(features))
    test_target = torch.from_numpy(np.array(target))

    # Load model
    traffic_prediction_path = dirname(interactive_app_path)
    saved_models_path = os.path.join(traffic_prediction_path, 'saved_models', 'last_saved_model.txt')
    with open(saved_models_path) as f:
        saved_model = f.read()

    latest_model_path = os.path.join(traffic_prediction_path, saved_model)
    checkpoint = torch.load(latest_model_path, map_location=torch.device('cpu'))
    model_stgcn = model.Stgcn_Model(checkpoint['model_nodes_num'], checkpoint['model_features_num'],
                                    checkpoint['model_input_timesteps'], checkpoint['model_num_output'])
    model_stgcn.load_state_dict(checkpoint['state_dict'])
    optimizer = optim.Adam(model_stgcn.parameters(), lr=checkpoint['model_lr'])
    optimizer = optimizer.load_state_dict(checkpoint['opti_state_dict'])
    loaded_model = model_stgcn
    loaded_optimizer = optimizer

    predicted = model_utils.predict(loaded_model, test_input, adj_mat)
    predicted_denorm = preprocessing_utils.denormalize(predicted, stds[0], means[0])

    return np.array(predicted_denorm), A, X, metadata
