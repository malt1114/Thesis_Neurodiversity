#Torch
import torch
from torch.optim.lr_scheduler import StepLR
from torch_geometric.loader import DataLoader
from torch_geometric import seed_everything


#other imports 
import numpy as np 
import random

#New imports
from help_funcs.param import get_optimizer, get_loss_function
from help_funcs.training import train_loop, val_loop
from help_funcs.data_func import get_node_features, load_dataset
from models.GCN import GCN



if __name__ =="__main__":
    import time

    #Get number of features + print their names
    selected_feature = ['var_bin', 'mean_bin']

    #Load data
    train_data = load_dataset(dataset = 'val', 
                              num_of_classes = 4,
                              feature_names = selected_feature,
                              edge_names = ['corr_var_bin_21','corr_mean_bin_21',
                                            'corr_var_bin_42','corr_mean_bin_42',
                                            'corr_var_bin_84','corr_mean_bin_84'],
                              edge_w_thres = None,
                              drop_strategy = 2, 
                              edge_w_abs = False, 
                              edge_w_relative = 0.3,
                              GAT = True)
    
    #for i in train_data:
    #    print(i.edge_attr)

    #import copy

    """for thres in [0.0, 0.2, 0.1, 0.4, 0.3]:
        train_data_copy = []
        for i in train_data:
            network = copy.deepcopy(i)
            network.edge_attr = torch.where(network.edge_attr >= thres, network.edge_attr, 0)
            train_data_copy.append(network)
        print(thres, torch.sum(train_data_copy[1].edge_attr))
    
    print('Next')

    for thres in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        train_data_copy = []
        for i in train_data:
            network = copy.deepcopy(i)
            network.edge_attr = torch.where(network.edge_attr >= thres, network.edge_attr, 0)
            train_data_copy.append(network)
        print(thres, torch.sum(train_data_copy[1].edge_attr))"""