#Torch
import torch
from torch.nn import Dropout
from torch_geometric.nn import (GCNConv, aggr, 
                                MessagePassing)
from torch_geometric import seed_everything
from torch_geometric.nn.norm import LayerNorm
from help_funcs.param import (get_activation, 
                              get_pool, 
                              get_last_out_function)

class ConvLayer(MessagePassing):
    def __init__(self, in_, out_, agg_func, activation, norm_):
        """The GCN convolutional layer

        Args:
            in_ (int): input feature size
            out_ (int): output feature size
            agg_func (_type_): the aggregation function (default is mean)
            activation (str): the activation function used
            norm_ (str): the type of normalization
        """
        super().__init__(agg_func)
        self.conv = GCNConv(in_, 
                            out_, 
                            add_self_loops = True,
                            normalize = True, 
                            bias = True)
        self.activation = get_activation(activation)
        self.norm = LayerNorm(out_, mode = norm_)


    def forward(self, x, edge_index, edge_weight):
        x = self.conv(x, edge_index, edge_weight)
        x = self.activation(x)
        x = self.norm(x)
        return x

class GCN(torch.nn.Module):
    def __init__(self, in_:int, out_:int, layer_1_out:int, dropout_rate:float, activation_:str, pool_:str, norm_:str, out_func_:str, random_seed:int):
        """Used to initialize the GCN model

        Args:
            in_ (int): the input shape of node features
            out_ (int): the number of features out
            layer_1_out (int): the size of hidden layer
            dropout_rate (float): dropout rate
            activation_ (str): the activation function
            pool_ (str): the pooling function
            norm_ (str): the normalization type
            out_func_ (str): the last activation function
            random_seed (int): the random seed
        """               
        super().__init__()
        #Set random seeds
        seed_everything(random_seed)

        self.conv1 = ConvLayer(in_ = in_, #number of features
                            out_ = layer_1_out, 
                            agg_func = aggr.MeanAggregation(),
                            activation = activation_,
                            norm_ = norm_)
        
        self.conv2 = ConvLayer(in_ = layer_1_out, 
                                out_ = out_, #number of classes
                                agg_func = aggr.MeanAggregation(),
                                activation = activation_,
                                norm_ = norm_)
        
        self.pool = get_pool(pool_)
        self.out_func = get_last_out_function(out_func_)
        self.dropout = Dropout(p = dropout_rate)


    def forward(self, x, edge_index, edge_weight, data):

        x = self.conv1(x, edge_index, edge_weight)
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_weight)
        
        #Get global value for network
        x = self.pool(x, data)
        x = self.out_func(x)
        
        return x