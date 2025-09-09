#Torch
import torch
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch_geometric import seed_everything

#other imports 
import wandb
import numpy as np 
import random
import argparse
import copy

#New imports
from help_funcs.param import get_optimizer, get_loss_function
from help_funcs.training import train_loop, val_loop
from help_funcs.data_func import get_node_features, load_dataset
from models.GCN import GCN

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def main():
    run = wandb.init()
    config = wandb.config

    seed_everything(config.random_seed)
    print('Device:', 'cuda:0'  if torch.cuda.is_available() else 'cpu', flush = True)

    device = torch.device('cuda:0'  if torch.cuda.is_available() else 'cpu')
    
    if config.num_of_classes == 2:
        num_of_classes = 1
        class_names = ['TD', 'Non-TD']
    else:
        num_of_classes = config.num_of_classes
        class_names = ['TD', 'ASD-ADHD', 'ASD', 'ADHD']
        
    model = GCN(in_ = config.num_of_features,
                out_ = num_of_classes,
                layer_1_out = config.hidden_channels_1, 
                dropout_rate = config.dropout, 
                activation_ = config.activation,
                pool_ = config.pool,
                norm_ = config.layer_norm,
                out_func_ = config.loss_func,
                random_seed = config.random_seed)
    
    model = model.to(device)
    loss_func = get_loss_function(config.loss_func).to(device)

    #work_seed
    g = torch.Generator()
    g.manual_seed(config.random_seed)
        
    #Make dataset loaders
    train_loader = DataLoader(train_dict[str(float(config.edge_weight_thres))], 
                              batch_size=config.batch_size, 
                              shuffle = True,
                              pin_memory=True,
                              worker_init_fn=seed_worker,
                              generator=g
                              )
                            
    val_loader = DataLoader(val_dict[str(float(config.edge_weight_thres))], 
                            batch_size=config.batch_size,
                            pin_memory=True,
                            worker_init_fn=seed_worker,
                            generator=g)

    # Optimizer
    optimizer = get_optimizer(config.optimizer, model, config.learning_rate)

    #Keep track of the best epoch
    min_val, min_train, max_f1, min_val_epoch = 10**6, 10**6, 10**6, -1
    min_val_labels, min_val_predictions = [], []


    #Train model
    for epoch in range(config.num_epochs):
        train_loss = train_loop(model, train_loader, optimizer, device, loss_func, loss_name = config.loss_func)
        val_loss, f1score, f1score_data = val_loop(model, val_loader, device, loss_func, loss_name = config.loss_func)

        #If best epoch, save variables
        if val_loss < min_val:
            min_val, min_train, max_f1, min_val_epoch = val_loss, train_loss, f1score, epoch
            min_val_labels, min_val_predictions = f1score_data[0], f1score_data[1]
        
        wandb.log({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "f1_score": f1score
        })

    wandb.log({
    "min_val_loss": min_val,
    "min_train_loss": min_train,
    "min_val_f1_score": max_f1,
    "min_val_epoch": min_val_epoch,
    "conf_mat_min_val" : wandb.plot.confusion_matrix(probs=None,
                                                    y_true=min_val_labels, preds=min_val_predictions,
                                                    class_names=class_names),
    })
    wandb.finish()

if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_of_classes', type=int, required=True)
    parser.add_argument('--feature_names', nargs='+', required=True)
    parser.add_argument('--edge_names', nargs='+', required=True)
    parser.add_argument('--absolute_weights', type=bool, required=True)
    parser.add_argument('--edge_threshold', type=float, nargs='+', required=True)
    parser.add_argument('--sweep_id', type=str, required=True)
    
    args = parser.parse_args()

    #Load data
    train_dict = {} 
    val_dict = {}

    for i in args.edge_threshold:
        train_data = load_dataset(dataset = 'train', 
                                num_of_classes = args.num_of_classes,
                                feature_names = args.feature_names,
                                edge_names = args.edge_names,
                                edge_w_thres = i,
                                drop_strategy = None,
                                edge_w_abs = False,
                                GAT = False
                                )
        
        train_dict[str(float(i))] = train_data
        print(train_dict.keys(), flush = True)

        val_data = load_dataset(dataset = 'val', 
                                num_of_classes = args.num_of_classes,
                                feature_names = args.feature_names,
                                edge_names = args.edge_names,
                                edge_w_thres = i,
                                drop_strategy = None,
                                edge_w_abs = False,
                                GAT = False
                                )
        val_dict[str(float(i))] = val_data

    #Make sweep
    wandb.agent(args.sweep_id, function=main)