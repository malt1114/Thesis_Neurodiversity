from torch.nn import (Softmax, 
                      ReLU, 
                      Sigmoid, 
                      LogSoftmax,
                      NLLLoss, 
                      BCELoss)
from torch import tensor
from torch.optim import (Adam, SGD)
from torch_geometric.nn import (global_mean_pool, 
                                global_max_pool)

def get_activation(func_name:str):
    if func_name == 'sigmoid':
        return Sigmoid()
    elif func_name == 'softmax':
        return Softmax(dim = 1)
    elif func_name == 'relu':
        return ReLU()
    else: 
        print('Activation func should be relu, softmax or sigmoid', flush = True)

def get_pool(func_name:str):
    if func_name == 'global_max_pool':
        return global_max_pool
    elif func_name == 'global_mean_pool':
        return global_mean_pool
    else: 
        print('Pool func should be global mean or max', flush = True)

def get_optimizer(func_name, model, learning_rate):
    # Optimizer
    if func_name == 'adam':
        return Adam(model.parameters(), lr=learning_rate)
    elif func_name == 'sgd':
        return SGD(model.parameters(), lr=learning_rate)
    else:
        print('The optimizer should be either adam or sgd', flush= True)

def get_last_out_function(func_name):
    if func_name == 'NLL_Loss':
        return LogSoftmax(dim=1)
    elif func_name == 'BCE':
        return Sigmoid()
    else:
        print("No such out function for:", func_name, flush= True)

def get_loss_function(func_name):
    if func_name == 'NLL_Loss':
        return NLLLoss()
    elif func_name == 'BCE':
        return BCELoss()
    else:
        print("No such loss function as:", func_name, flush= True)