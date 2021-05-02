import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class Temporal_Layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel):
        super(Temporal_Layer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel))
        self.conv2 = nn.Conv2d(in_channels, out_channels, (1, kernel))
        
    def forward(self, x):
        x = x.permute(0,3,1,2)
        normal = self.conv1(x)
        sig = torch.sigmoid(self.conv2(x))
        out = normal * sig
        out = out.permute(0,2,3,1)
        # Convert back from NCHW to NHWC
        return out
        
        
class Stgcn_Block(nn.Module):
    def __init__(self, in_channels, spatial_channels, out_channels, nodes_num):
        super(Stgcn_Block, self).__init__()
        self.temporal_layer1 = Temporal_Layer(in_channels, out_channels, kernel = 2) 
        self.temporal_layer2 = Temporal_Layer(in_channels = spatial_channels, out_channels = out_channels, kernel = 2)
        
        self.weight = nn.Parameter(torch.FloatTensor(out_channels, spatial_channels)) 
        self.initialise_weight()
        
        self.batch_norm = nn.BatchNorm2d(nodes_num)
        
    def initialise_weight(self):
        std_dv = 1. / math.sqrt(self.weight.shape[1])
        self.weight.data.uniform_(-std_dv, std_dv)
        
    def forward(self, x, adj_hat):
        # First temporal Block
        temporal_block1 = self.temporal_layer1(x) #temporal_block1.shape = batch_size, num_nodes, input_timesteps -1, out_channels
        
        #Spatial Graph Convolution
        #n = batch_size, h = nodes_num, w = input_timesteps -1, c = out_channels
        t = temporal_block1.permute(1,0,2,3) #Converts tensor from nhwc to hnwc for multiplication with adj_matrix
        gconv1 = torch.einsum("ij, jklm -> kilm", [adj_hat, t]) #(h,h) * (h,n,w,c) -> (n,h,w,c) 
        gconv2 = F.relu(torch.matmul(gconv1, self.weight)) #gconv2.shape = n,h,w,c* where c* = spatial_channels
        
        #Second Temporal Block
        temporal_block2 = self.temporal_layer2(gconv2) #temporal_block2.shape = batch_size, num_nodes, input_timesteps -2, out_channels
        
        out = self.batch_norm(temporal_block2)
        return out

class Stgcn_Model(nn.Module):
    def __init__(self, nodes_num, features_num, input_timesteps, num_output):
        super(Stgcn_Model, self).__init__()
        self.stgcn_block1 = Stgcn_Block(in_channels = features_num, spatial_channels = 16, out_channels = 64,
                                       nodes_num = nodes_num)
        
        self.stgcn_block2 = Stgcn_Block(in_channels = 64, spatial_channels = 16,  out_channels = 64,
                                       nodes_num = nodes_num)
        
        self.temporal_layer = Temporal_Layer(in_channels = 64, out_channels = 64, kernel = 2)
        self.fc = nn.Linear((input_timesteps - 5) * 64, num_output)

    def forward(self, adj_hat, x):
        #x.shape = batch_size, num_nodes, input_timesteps, features_num
        
        out1 = self.stgcn_block1(x, adj_hat) 
        #out1.shape = batch_size, num_nodes, input_timesteps - 2, out_channels 
        
        out2 = self.stgcn_block2(out1, adj_hat)
        #out2.shape = batch_size, num_nodes, input_timesteps -4, out_channels
            
        out3 = self.temporal_layer(out2)
        #out3.shape = batch_size, num_nodes, input_timesteps -5, out_channels
        
        out3_reshaped = out3.reshape(out3.shape[0], out3.shape[1], -1)
        
        out4 = self.fc(out3_reshaped)
        
        return out4
    