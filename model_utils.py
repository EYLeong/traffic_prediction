import time
import os
import math

import torch
from torch import optim, nn

import model

def predict(stgcn, x_input, adj_mat):
    with torch.no_grad():
        stgcn.eval()
        out = stgcn(adj_mat, x_input)
    return out

def validate(stgcn, loss_criterion, val_input, val_target, adj_mat, batch_size, device):
    num_samples = val_input.shape[0]
    shuffled_order = torch.randperm(num_samples)
    total_val_loss = 0
    counter = 0
    
    stgcn.eval()
    
    for i in range(math.ceil(num_samples / batch_size)):
        start = i * batch_size
        batch = shuffled_order[start:start+batch_size]
        
        val_input_batch = val_input[batch].to(device = device)
        val_target_batch = val_target[batch].to(device = device)
        out = stgcn(adj_mat, val_input_batch)
        val_loss = loss_criterion(out, val_target_batch).to(device='cpu')
        total_val_loss += val_loss
        counter += 1
        
    total_val_loss = total_val_loss / counter
    return total_val_loss

def save_model(model, optimizer):
    
    os.environ['TZ'] = 'Singapore'
    time.tzset()
    date = time.strftime("%Y%m%d")
    timestamp = time.strftime("%H_%M_%S")
    if not os.path.exists('./saved_models/' + date):
        os.makedirs('./saved_models/' + date)
    path = './saved_models/' + date + '/' + timestamp
    
    checkpoint = {'state_dict': model.state_dict(),
                  'opti_state_dict': optimizer.state_dict(),
                  'model_lr': model.lr,
                  'model_nodes_num': model.nodes_num,
                  'model_features_num': model.features_num,
                  'model_input_timesteps': model.input_timesteps,
                  'model_num_output': model.num_output
                  }
    
    torch.save(checkpoint, path)
    
    f = open("./saved_models/last_saved_model.txt", "w")
    f.write(path)
    f.close()
    print(f"Model has been saved to path : {path}")
    
def load_model(path=None):
    
    if path == None:
        with open("./saved_models/last_saved_model.txt") as f:
            path = f.read()
    
    print(f"Loading model in path : {path}")
    
    checkpoint = torch.load(path)
    model_stgcn = model.Stgcn_Model(checkpoint['model_nodes_num'], checkpoint['model_features_num'], checkpoint['model_input_timesteps'], checkpoint['model_num_output'])
    model_stgcn.load_state_dict(checkpoint['state_dict'])
    optimizer = optim.Adam(model_stgcn.parameters(), lr=checkpoint['model_lr'])
    optimizer = optimizer.load_state_dict(checkpoint['opti_state_dict'])
    
    return model_stgcn, optimizer