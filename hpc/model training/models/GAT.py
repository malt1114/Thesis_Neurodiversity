#Torch
import torch
from torch.nn import Dropout, Linear, ReLU
from torch_geometric.nn import (GATConv, aggr, 
                                MessagePassing)
from torch_geometric import seed_everything
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn.norm import LayerNorm
from help_funcs.param import (get_activation, 
                              get_pool, 
                              get_last_out_function)

class AttentionConvLayer(MessagePassing):
    def __init__(self, in_, out_, agg_func, activation, heads_, concat_, norm_):
        super().__init__(agg_func)
        self.conv = GATConv(in_, 
                            out_, 
                            add_self_loops = True,
                            concat = concat_,
                            edge_dim = 6,
                            heads = heads_)
        self.activation = get_activation(activation)
        norm_dim = out_ * heads_ if concat_ else out_
        self.norm = LayerNorm(norm_dim, mode = norm_)

    def forward(self, x, edge_index, edge_attr):
        x = self.conv(x, edge_index, edge_attr)
        x = self.activation(x)        
        x = self.norm(x)
        return x

class GAT(torch.nn.Module):
    def __init__(self, in_, out_, layer_1_out, dropout_rate, activation_, pool_, norm_, out_func_, random_seed, heads_):
        super().__init__()
        #Set random seeds
        seed_everything(random_seed)
        #set_seed(seed = random_seed)

        self.conv1 = AttentionConvLayer(in_ = in_, #number of features
                                        out_ = layer_1_out, 
                                        agg_func = aggr.MeanAggregation(),
                                        activation = activation_,
                                        norm_ = norm_,
                                        heads_ = heads_,
                                        concat_ = True)
        
        self.conv2 = AttentionConvLayer(in_ = layer_1_out*heads_, 
                                        out_ = out_, #number of classes
                                        agg_func = aggr.MeanAggregation(),
                                        activation = activation_, 
                                        norm_ = norm_,
                                        heads_ = 1,
                                        concat_ = True)
        
        self.pool = get_pool(pool_)
        self.out_func = get_last_out_function(out_func_)
        self.dropout = Dropout(p = dropout_rate)

    def forward(self, x, edge_index, edge_attr, data):

        x = self.conv1(x, edge_index, edge_attr)
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_attr)
        
        #Get global value for network
        x = self.pool(x, data)
        x = self.out_func(x)
        
        return x
    
class GATFLAT(torch.nn.Module):
    def __init__(self, in_, out_, layer_1_out, dropout_rate, activation_, norm_, out_func_, random_seed, heads_):
        super().__init__()
        #Set random seeds
        seed_everything(random_seed)
        #set_seed(seed = random_seed)

        self.conv1 = AttentionConvLayer(in_ = in_, #number of features
                                        out_ = layer_1_out, 
                                        agg_func = aggr.MeanAggregation(),
                                        activation = activation_,
                                        norm_ = norm_,
                                        heads_ = heads_,
                                        concat_ = True)
        
        self.dropout = Dropout(p = dropout_rate)

        self.conv2 = AttentionConvLayer(in_ = layer_1_out*heads_, 
                                        out_ = layer_1_out, #number of classes
                                        agg_func = aggr.MeanAggregation(),
                                        activation = activation_, 
                                        norm_ = norm_,
                                        heads_ = heads_,
                                        concat_ = True)
        
        self.nn_layer = Linear(layer_1_out*heads_*17, out_)
        
        self.out_func = get_last_out_function(out_func_)

    def forward(self, x, edge_index, edge_attr, data):
        x = self.conv1(x, edge_index, edge_attr)
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_attr)
        #reshape to fit linear layer
        x, mask = to_dense_batch(x, data)
        x = x.reshape(x.shape[0], -1)
        x = self.nn_layer(x)
        x = self.out_func(x)
        
        return x

class GATFLAT_MC_DROPOUT(torch.nn.Module):
    def __init__(self, in_, out_, layer_1_out, dropout_rate, activation_, norm_, out_func_, random_seed, heads_):
        super().__init__()
        #Set random seeds
        seed_everything(random_seed)
        #set_seed(seed = random_seed)

        self.conv1 = AttentionConvLayer(in_ = in_, #number of features
                                        out_ = layer_1_out, 
                                        agg_func = aggr.MeanAggregation(),
                                        activation = activation_,
                                        norm_ = norm_,
                                        heads_ = heads_,
                                        concat_ = True)
        
        self.dropout = Dropout(p = dropout_rate)

        self.conv2 = AttentionConvLayer(in_ = layer_1_out*heads_, 
                                        out_ = layer_1_out, #number of classes
                                        agg_func = aggr.MeanAggregation(),
                                        activation = activation_, 
                                        norm_ = norm_,
                                        heads_ = heads_,
                                        concat_ = True)
        
        self.nn_layer = Linear(layer_1_out*heads_*17, out_)
        
        self.out_func = get_last_out_function(out_func_)

    def forward(self, x, edge_index, edge_attr, data):
        x = self.conv1(x, edge_index, edge_attr)
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = self.dropout(x)
        #reshape to fit linear layer
        x, mask = to_dense_batch(x, data)
        x = x.reshape(x.shape[0], -1)
        x = self.nn_layer(x)
        x = self.out_func(x)
        
        return x