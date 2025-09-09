#Torch
import torch
from torch_geometric.loader import DataLoader
from torch_geometric import seed_everything

#other imports 
import wandb
import numpy as np 
import random
import argparse

#New imports
from help_funcs.param import get_optimizer, get_loss_function
from help_funcs.training import train_loop, val_loop
from help_funcs.data_func import load_dataset
from models.GAT import GATFLAT

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def run_gat(config):
    run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="malthe-pabst-it-universitetet-i-k-benhavn",
        # Set the wandb project where this run will be logged.
        project="Final Network Models",
        # Track hyperparameters and run metadata.
        config= config)
    
    config = run.config

    print('Device:', 'cuda:0'  if torch.cuda.is_available() else 'cpu', flush = True)

    device = torch.device('cuda:0'  if torch.cuda.is_available() else 'cpu')
    
    train_data = load_dataset(dataset = 'train', 
                              num_of_classes = config.num_of_classes,
                              feature_names = config.feature_names,
                              edge_names = config.edge_names,
                              edge_w_thres = config.edge_w_thres,
                              drop_strategy = config.drop_strategy,
                              edge_w_abs = config.edge_w_abs,
                              GAT=True)
    
    val_data = load_dataset(dataset = 'val', 
                            num_of_classes = config.num_of_classes,
                            feature_names = config.feature_names,
                            edge_names = config.edge_names,
                            edge_w_thres = config.edge_w_thres,
                            drop_strategy = config.drop_strategy,
                            edge_w_abs = config.edge_w_abs,
                            GAT=True)
    
    seed_everything(config.random_seed)


    if config.num_of_classes == 2:
        num_of_classes = 1
        class_names = ['TD', 'Non-TD']
    else:
        num_of_classes = config.num_of_classes
        class_names = ['TD', 'ASD-ADHD', 'ASD', 'ADHD']
        
    model = GATFLAT(in_ = config.num_of_features,
                    out_ = num_of_classes,
                    layer_1_out = config.hidden_channels_1, 
                    dropout_rate = config.dropout, 
                    activation_ = config.activation,
                    norm_ = config.layer_norm,
                    out_func_ = config.loss_name,
                    random_seed = config.random_seed,
                    heads_ = config.att_heads)
    
    model = model.to(device)
    loss_func = get_loss_function(config.loss_name).to(device)

    #work_seed
    g = torch.Generator()
    g.manual_seed(config.random_seed)
    
    #Make dataset loaders
    train_loader = DataLoader(train_data, 
                              batch_size=config.batch_size, 
                              shuffle = True,
                              pin_memory=True,
                              worker_init_fn=seed_worker,
                              generator=g
                              )

    val_loader = DataLoader(val_data, 
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
        train_loss = train_loop(model, train_loader, optimizer, device, loss_func, loss_name = config.loss_name)
        val_loss, f1score, f1score_data = val_loop(model, val_loader, device, loss_func, loss_name = config.loss_name)

        #If best epoch, save variables
        if val_loss < min_val:
            min_val, min_train, max_f1, min_val_epoch = val_loss, train_loss, f1score, epoch
            min_val_labels, min_val_predictions = f1score_data[0], f1score_data[1]
            #Save and upload the model to WANDB
            model_path = f'saved_models/{config.MODEL_NAME}_{epoch + 1}.pt'
            torch.save(model.state_dict(), model_path)
            wandb.save(model_path)
        
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
   
    ########################################
    # GAT Multi: no absolute values or threshold #
    ########################################
    
    run_gat(config = { "model_type": 'GAT',
                        "MODEL_NAME": "GAT_Multi_no_absolute_values_or_threshold",
                        "hardware": 'GPU',
                        "activation": "relu", 
                        "att_heads": 4, 
                        "batch_size": 16, 
                        "dropout": 0.1, 
                        "hidden_channels_1": 16, 
                        "num_of_classes": 4, 
                        "num_epochs": 500, 
                        "num_of_features": 16, 
                        "layer_norm": 'graph', 
                        "learning_rate": 0.002 , 
                        "loss_name": "NLL_Loss", 
                        "optimizer": "sgd", 
                        "random_seed": 42,
                        "feature_names": ["var_bin", "mean_bin"],
                        "edge_names": ["corr_var_bin_21", "corr_mean_bin_21", "corr_var_bin_42",
                                       "corr_mean_bin_42", "corr_var_bin_84", "corr_mean_bin_84"],
                        "drop_strategy": None,
                        "edge_w_thres": None, 
                        "edge_w_abs": False,
                        })
    
    ########################################
    # GAT Multi: absolute values and threshold #
    ########################################
    
    run_gat(config = { "model_type": 'GAT',
                      "MODEL_NAME": "GAT_Multi_absolute_value_and_threshold",
                        "hardware": 'GPU',
                        "activation": "relu", 
                        "att_heads": 16, 
                        "batch_size": 16, 
                        "dropout": 0, 
                        "hidden_channels_1": 8, 
                        "num_of_classes": 4, 
                        "num_epochs": 500, 
                        "num_of_features": 16, 
                        "layer_norm": 'graph', 
                        "learning_rate": 0.002 , 
                        "loss_name": "NLL_Loss", 
                        "optimizer": "sgd", 
                        "random_seed": 42,
                        "feature_names": ["var_bin", "mean_bin"],
                        "edge_names": ["corr_var_bin_21", "corr_mean_bin_21", "corr_var_bin_42",
                                       "corr_mean_bin_42", "corr_var_bin_84", "corr_mean_bin_84"],
                        "drop_strategy": None,
                        "edge_w_thres": 0.7, 
                        "edge_w_abs": True,
                        })
    
    ############################################
    # GAT Multi: absolute values and drop strategy 2 #
    ############################################
    
    run_gat(config = { "model_type": 'GAT',
                       "MODEL_NAME": "GAT_Multi_absolute_value_and_strategy_2",
                        "hardware": 'GPU',
                        "activation": "relu", 
                        "att_heads": 4, 
                        "batch_size": 16, 
                        "dropout": 0.2, 
                        "hidden_channels_1": 128, 
                        "num_of_classes": 4, 
                        "num_epochs": 500, 
                        "num_of_features": 16, 
                        "layer_norm": 'graph', 
                        "learning_rate":  0.001, 
                        "loss_name": "NLL_Loss", 
                        "optimizer": "sgd", 
                        "random_seed": 42,
                        "feature_names": ["var_bin", "mean_bin"],
                        "edge_names": ["corr_var_bin_21", "corr_mean_bin_21", "corr_var_bin_42",
                                       "corr_mean_bin_42", "corr_var_bin_84", "corr_mean_bin_84"],
                        "drop_strategy": 2,
                        "edge_w_thres": 0.2, 
                        "edge_w_abs": True,
                        })
    
    ############################################
    # GAT Multi: absolute values and drop strategy 3 #
    ############################################
    
    run_gat(config = { "model_type": 'GAT',
                        "MODEL_NAME": "GAT_Multi_absolute_value_and_strategy_3",
                        "hardware": 'GPU',
                        "activation": "relu", 
                        "att_heads": 4, 
                        "batch_size": 16, 
                        "dropout": 0.2, 
                        "hidden_channels_1": 128, 
                        "num_of_classes": 4, 
                        "num_epochs": 500, 
                        "num_of_features": 16, 
                        "layer_norm": 'graph', 
                        "learning_rate": 0.001, 
                        "loss_name": "NLL_Loss", 
                        "optimizer": "sgd", 
                        "random_seed": 42,
                        "feature_names": ["var_bin", "mean_bin"],
                        "edge_names": ["corr_var_bin_21", "corr_mean_bin_21", "corr_var_bin_42",
                                       "corr_mean_bin_42", "corr_var_bin_84", "corr_mean_bin_84"],
                        "drop_strategy": 3,
                        "edge_w_thres": 0.3, 
                        "edge_w_abs": True,
                        })
    
    ############################################
    # GAT Multi: absolute values and drop strategy 4 #
    ############################################
    
    run_gat(config = { "model_type": 'GAT',
                       "MODEL_NAME": "GAT_Multi_absolute_value_and_strategy_4",
                        "hardware": 'GPU',
                        "activation": "relu", 
                        "att_heads": 4, 
                        "batch_size": 16, 
                        "dropout": 0.2, 
                        "hidden_channels_1": 128, 
                        "num_of_classes": 4, 
                        "num_epochs": 500, 
                        "num_of_features": 16, 
                        "layer_norm": 'graph', 
                        "learning_rate": 0.001, 
                        "loss_name": "NLL_Loss", 
                        "optimizer": "sgd", 
                        "random_seed": 42,
                        "feature_names": ["var_bin", "mean_bin"],
                        "edge_names": ["corr_var_bin_21", "corr_mean_bin_21", "corr_var_bin_42",
                                       "corr_mean_bin_42", "corr_var_bin_84", "corr_mean_bin_84"],
                        "drop_strategy": 4,
                        "edge_w_thres": 0.6, 
                        "edge_w_abs": True,
                        })

    ########################################
    # GAT Binary: no absolute values or threshold #
    ########################################
    
    run_gat(config = { "model_type": 'GAT',
                       "MODEL_NAME": "GAT_Binary_no_absolute_values_or_threshold",
                        "hardware": 'GPU',
                        "activation": "relu", 
                        "att_heads": 4, 
                        "batch_size": 16, 
                        "dropout": 0.1, 
                        "hidden_channels_1": 64, 
                        "num_of_classes": 2, 
                        "num_epochs": 500, 
                        "num_of_features": 16, 
                        "layer_norm": 'graph', 
                        "learning_rate": 0.002 , 
                        "loss_name": "BCE", 
                        "optimizer": "sgd", 
                        "random_seed": 42,
                        "feature_names": ["var_bin", "mean_bin"],
                        "edge_names": ["corr_var_bin_21", "corr_mean_bin_21", "corr_var_bin_42",
                                       "corr_mean_bin_42", "corr_var_bin_84", "corr_mean_bin_84"],
                        "drop_strategy": None,
                        "edge_w_thres": None, 
                        "edge_w_abs": False,
                        })
    
    ########################################
    # GAT Binary: absolute values and threshold #
    ########################################
    
    run_gat(config = { "model_type": 'GAT',
                      "MODEL_NAME": "GAT_Binary_absolute_value_and_threshold",
                        "hardware": 'GPU',
                        "activation": "relu", 
                        "att_heads": 4, 
                        "batch_size": 16, 
                        "dropout": 0.1, 
                        "hidden_channels_1": 64, 
                        "num_of_classes": 2, 
                        "num_epochs": 500, 
                        "num_of_features": 16, 
                        "layer_norm": 'graph', 
                        "learning_rate": 0.002 , 
                        "loss_name": "BCE", 
                        "optimizer": "sgd", 
                        "random_seed": 42,
                        "feature_names": ["var_bin", "mean_bin"],
                        "edge_names": ["corr_var_bin_21", "corr_mean_bin_21", "corr_var_bin_42",
                                       "corr_mean_bin_42", "corr_var_bin_84", "corr_mean_bin_84"],
                        "drop_strategy": None,
                        "edge_w_thres": 0.2, 
                        "edge_w_abs": True,
                        })
    
    ###################################################
    # GAT Binary: absolute values and drop strategy 2 #
    ###################################################
    
    run_gat(config = { "model_type": 'GAT',
                       "MODEL_NAME": "GAT_Binary_absolute_value_and_strategy_2",
                        "hardware": 'GPU',
                        "activation": "relu", 
                        "att_heads": 2, 
                        "batch_size": 16, 
                        "dropout": 0, 
                        "hidden_channels_1": 128, 
                        "num_of_classes": 2, 
                        "num_epochs": 500, 
                        "num_of_features": 16, 
                        "layer_norm": 'graph', 
                        "learning_rate": 0.005, 
                        "loss_name": "BCE", 
                        "optimizer": "sgd", 
                        "random_seed": 42,
                        "feature_names": ["var_bin", "mean_bin"],
                        "edge_names": ["corr_var_bin_21", "corr_mean_bin_21", "corr_var_bin_42",
                                       "corr_mean_bin_42", "corr_var_bin_84", "corr_mean_bin_84"],
                        "drop_strategy": 2,
                        "edge_w_thres": 0.3, 
                        "edge_w_abs": True,
                        })
    
    ###################################################
    # GAT Binary: absolute values and drop strategy 3 #
    ###################################################
    
    run_gat(config = { "model_type": 'GAT',
                       "MODEL_NAME": "GAT_Binary_absolute_value_and_strategy_3",
                        "hardware": 'GPU',
                        "activation": "relu", 
                        "att_heads": 2, 
                        "batch_size": 16, 
                        "dropout": 0, 
                        "hidden_channels_1": 64, 
                        "num_of_classes": 2, 
                        "num_epochs": 500, 
                        "num_of_features": 16, 
                        "layer_norm": 'graph', 
                        "learning_rate": 0.01, 
                        "loss_name": "BCE", 
                        "optimizer": "sgd", 
                        "random_seed": 42,
                        "feature_names": ["var_bin", "mean_bin"],
                        "edge_names": ["corr_var_bin_21", "corr_mean_bin_21", "corr_var_bin_42",
                                       "corr_mean_bin_42", "corr_var_bin_84", "corr_mean_bin_84"],
                        "drop_strategy": 3,
                        "edge_w_thres": 0.4, 
                        "edge_w_abs": True,
                        })
    
    ###################################################
    # GAT Binary: absolute values and drop strategy 4 #
    ###################################################
    
    run_gat(config = { "model_type": 'GAT',
                        "MODEL_NAME": "GAT_Binary_absolute_value_and_strategy_4",
                        "hardware": 'GPU',
                        "activation": "relu", 
                        "att_heads": 4, 
                        "batch_size": 16, 
                        "dropout": 0, 
                        "hidden_channels_1": 8, 
                        "num_of_classes": 2, 
                        "num_epochs": 500, 
                        "num_of_features": 16, 
                        "layer_norm": 'graph', 
                        "learning_rate": 0.01, 
                        "loss_name": "BCE", 
                        "optimizer": "sgd", 
                        "random_seed": 42,
                        "feature_names": ["var_bin", "mean_bin"],
                        "edge_names": ["corr_var_bin_21", "corr_mean_bin_21", "corr_var_bin_42",
                                       "corr_mean_bin_42", "corr_var_bin_84", "corr_mean_bin_84"],
                        "drop_strategy": 4,
                        "edge_w_thres": 0.7, 
                        "edge_w_abs": True,
                        })













