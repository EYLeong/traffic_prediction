import time
import os
import math
import torch
from torch import optim, nn


def load_model(path=None, map_location=None):
    if path == None:
        saved_models_path = os.path.join(dirname(os.getcwd()), 'saved_models', 'last_saved_model.txt')
        with open(saved_models_path) as f:
            saved_model = f.read()

    print(f"Loading model in path : {path}")
    latest_model_path = os.path.join(dirname(os.getcwd()), saved_model)
    checkpoint = torch.load(latest_model_path, map_location=map_location)
    model_stgcn = model.Stgcn_Model(checkpoint['model_nodes_num'], checkpoint['model_features_num'],
                                    checkpoint['model_input_timesteps'], checkpoint['model_num_output'])
    model_stgcn.load_state_dict(checkpoint['state_dict'])
    optimizer = optim.Adam(model_stgcn.parameters(), lr=checkpoint['model_lr'])
    optimizer = optimizer.load_state_dict(checkpoint['opti_state_dict'])

    return model_stgcn, optimizer