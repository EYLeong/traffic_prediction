import time
import os, math, copy

import torch
from torch import optim, nn

from tqdm.notebook import tqdm

import model

def predict(model, x_input, adj_mat):
    with torch.no_grad():
        model.eval()
        out = model(adj_mat, x_input)
    return out

def validate(model, loss_criterion, val_input, val_target, adj_mat, batch_size):
    num_samples = val_input.shape[0]
    shuffled_order = torch.randperm(num_samples)
    total_val_loss = 0
    counter = 0
    
    model.eval()
    
    for i in range(math.ceil(num_samples / batch_size)):
        start = i * batch_size
        batch = shuffled_order[start:start+batch_size]
        
        val_input_batch = val_input[batch]
        val_target_batch = val_target[batch]
        out = model(adj_mat, val_input_batch)
        val_loss = loss_criterion(out, val_target_batch)
        total_val_loss += val_loss
        counter += 1
        
    total_val_loss = total_val_loss / counter
    return total_val_loss.item()

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
    
    
def save_model_timesteps(model, optimizer, input_timesteps, output_timesteps, loss):
    if not os.path.exists('./saved_models/timesteps'):
        os.makedirs('./saved_models/timesteps')
    path = "./saved_models/timesteps/input{}_output{}_loss{}".format(input_timesteps, output_timesteps, loss)
    checkpoint = {'state_dict': model.state_dict(),
              'opti_state_dict': optimizer.state_dict(),
              'model_lr': model.lr,
              'model_nodes_num': model.nodes_num,
              'model_features_num': model.features_num,
              'model_input_timesteps': model.input_timesteps,
              'model_num_output': model.num_output
              }
    torch.save(checkpoint, path)
    return path
    
def load_model(path=None, map_location=None):
    
    if path == None:
        with open("./saved_models/last_saved_model.txt") as f:
            path = f.read()
    
    print(f"Loading model in path : {path}")
    
    checkpoint = torch.load(path, map_location=map_location)
    model_stgcn = model.Stgcn_Model(checkpoint['model_nodes_num'], checkpoint['model_features_num'], checkpoint['model_input_timesteps'], checkpoint['model_num_output'])
    model_stgcn.load_state_dict(checkpoint['state_dict'])
    optimizer = optim.Adam(model_stgcn.parameters(), lr=checkpoint['model_lr'])
    optimizer = optimizer.load_state_dict(checkpoint['opti_state_dict'])
    
    return model_stgcn, optimizer

def train_epoch(model, optimizer, loss_criterion, adj_mat, x_input, x_target, batch_size):
    """
    Train function per epoch
    """
    
    model.train()
    
    num_samples = x_input.shape[0]
    shuffled_order = torch.randperm(num_samples)
    
    training_loss = []
    
    for i in range(math.ceil(num_samples / batch_size)):

        optimizer.zero_grad()
        
        start = i * batch_size
        batch = shuffled_order[start:start+batch_size]
        
        x_batch = x_input[batch]
        y_batch = x_target[batch]
        
        out = model(adj_mat, x_batch)
        loss = loss_criterion(out, y_batch)
        loss.backward()
        optimizer.step()
        
        training_loss.append(loss.item())
        
    return sum(training_loss) / len(training_loss)

def train(model, optimizer, lr, loss_criterion, epochs, patience, adj_mat, x_input, x_target, val_input, val_target, batch_size):
    best_loss = float("inf")
    early_stop = 0
    best_weights = None

    training_loss = []
    validation_loss = []

    pbar = tqdm(range(epochs))
    for epoch in pbar:

        pbar.set_description(f"Epoch {epoch}")

        loss = train_epoch(model, optimizer, loss_criterion, adj_mat, x_input, x_target, batch_size)
        training_loss.append(loss)

        with torch.no_grad():
            val_loss = validate(model, loss_criterion, val_input, val_target, adj_mat, batch_size)
            validation_loss.append(val_loss)

        pbar.set_postfix(training_loss=loss, validation_loss=val_loss)

        if val_loss < best_loss:
            early_stop = 0
            best_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
        else:
            early_stop += 1

        if early_stop == patience:
            model.load_state_dict(best_weights)
            break


    #For Model saving purposes
    model.lr = lr 
    model.nodes_num = adj_mat.shape[0]
    model.features_num = x_input.shape[3]
    model.input_timesteps = x_input.shape[2]
    model.num_output = x_target.shape[2]
    return model, training_loss, validation_loss