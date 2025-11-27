from torch.nn import (Softmax, 
                      ReLU, 
                      Sigmoid, 
                      LogSoftmax,
                      NLLLoss, 
                      BCELoss)
from torch.optim import (Adam, SGD)
from torch_geometric.nn import (global_mean_pool, 
                                global_max_pool)

def get_activation(func_name:str):
    """Returns the activation function

    Args:
        func_name (str): activation name

    Returns:
        _type_: the activation function
    """
    if func_name == 'sigmoid':
        return Sigmoid()
    elif func_name == 'softmax':
        return Softmax(dim = 1)
    elif func_name == 'relu':
        return ReLU()
    else: 
        print('Activation func should be relu, softmax or sigmoid', flush = True)

def get_pool(func_name:str):
    """Get pooling function

    Args:
        func_name (str): the function to use

    Returns:
        _type_: the pooling function
    """
    if func_name == 'global_max_pool':
        return global_max_pool
    elif func_name == 'global_mean_pool':
        return global_mean_pool
    else: 
        print('Pool func should be global mean or max', flush = True)

def get_optimizer(func_name:str, model, learning_rate:float):
    """Returns the optimizer

    Args:
        func_name (str): the optimizer to use
        model (_type_): the PyG model
        learning_rate (float): the learning rate

    Returns:
        _type_: the optimizer
    """
    # Optimizer
    if func_name == 'adam':
        return Adam(model.parameters(), lr=learning_rate)
    elif func_name == 'sgd':
        return SGD(model.parameters(), lr=learning_rate)
    else:
        print('The optimizer should be either adam or sgd', flush= True)

def get_last_out_function(func_name:str):
    """The last activation function to use, depending on if multiclass or not

    Args:
        func_name (str): the loss function name

    Returns:
        _type_: the activation function
    """
    if func_name == 'NLL_Loss':
        return LogSoftmax(dim=1)
    elif func_name == 'BCE':
        return Sigmoid()
    else:
        print("No such out function for:", func_name, flush= True)

def get_loss_function(func_name):
    """Get the loss function

    Args:
        func_name (str): the name of the loss function

    Returns:
        _type_: the loss functions
    """
    if func_name == 'NLL_Loss':
        return NLLLoss()
    elif func_name == 'BCE':
        return BCELoss()
    else:
        print("No such loss function as:", func_name, flush= True)