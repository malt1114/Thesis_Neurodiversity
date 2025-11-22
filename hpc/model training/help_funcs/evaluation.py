import yaml
import torch
import numpy as np
import pandas as pd
import seaborn as sns
from collections import Counter
import matplotlib.pyplot as plt

#Model related imports
from sklearn.metrics import roc_curve, auc, f1_score, confusion_matrix, accuracy_score
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import PCA
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch

#Model classes
from models.GCN import GCN
from models.GAT import GATFLAT, GATFLAT_MC_DROPOUT, GATFLAT_PCA

#Other functions
from help_funcs.data_func import load_dataset
from help_funcs.param import get_loss_function

def make_interference(data, model, loss_name, loss_func, batch_size):
    """Makes the interferencem and returns the scores, predictions and
       softmax values

    Args:
        data (_type_): the data loader
        model (_type_): the PyG model
        loss_name (_type_): the name of the loss function
        loss_func (_type_): the loss function
        batch_size (_type_): the batch size

    Returns:
        different information: loss, f1-score, accuracy, labels, predictions, extended labels, softmax_data 
    """

    val_loss = 0
    all_predictions = []
    extended_dia = []
    all_labels = []
    y_score = []
    
    for batch in data:
        x, edge_index, edge_weight, structure = batch.x, batch.edge_index, batch.edge_attr, batch.batch
        y_hat = model.forward(x, edge_index, edge_weight, structure)
        
        if loss_name != 'BCE':
            y_score += y_hat.exp().tolist()
        else:
            y_score += y_hat.tolist()

        y = batch.y
        if loss_name == 'BCE':
            y = y.float()
            y_hat = y_hat.squeeze(1)
        y = y

        loss = loss_func(y_hat, y)

        val_loss += (loss)*(len(batch)/batch_size)


        if loss_name == 'BCE':
            all_predictions += y_hat.round().tolist()
        else:
            all_predictions += torch.argmax(y_hat, dim=-1).tolist()
        
        all_labels += batch.y.tolist()

        extended_dia += batch.extended_dia

    if loss_name != 'BCE':
        class_labels = {0:'TD', 1:'ASD+ADHD', 2:'ASD', 3:'ADHD'}
    else: 
        class_labels = {0:'TD', 1:'Not-TD'}

    softmax_data = pd.DataFrame()
    softmax_data['label'] = all_labels
    softmax_data['label'] = softmax_data['label'].replace(class_labels)
    softmax_data['predicted'] = all_predictions
    softmax_data['predicted'] = softmax_data['predicted'].replace(class_labels)
    softmax_data['softmax_values'] = y_score
    
    if len(class_labels) != 2:
        for key, value in class_labels.items():
                softmax_data[value] = softmax_data['softmax_values'].apply(lambda x: x[key])

    return [val_loss/len(data), 
            f1_score(all_labels, all_predictions, average='macro'), 
            accuracy_score(all_labels, all_predictions),
            y_score, 
            all_labels, 
            all_predictions, 
            extended_dia,
            softmax_data]

def plot_ruc(roc_curve_data, num_of_classes, fig_name = None):
    """Plots the ruc curves

    Args:
        roc_curve_data (dict): data for each class and their softmax values
        num_of_classes (int): number of classes
        fig_name (_type_, optional): the figure name used when saved. Defaults to None.
    """
    # Define class labels and split names
    split_names = ['train', 'val', 'test']
    colors = {0:'#C66526', 1:'#469C76', 2:'#D39233', 3:'#3171AD'}

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))
    if num_of_classes == 4:
        class_labels_legends = {0:'TD', 1:'ASD+ADHD', 2:'ASD', 3:'ADHD'}
    else: 
        class_labels_legends = {0:'TD', 1:'Non-TD'}

    for ax, split_name in zip(axes, split_names):
        y_true = np.array(roc_curve_data[split_name][0])
        y_probs = np.array(roc_curve_data[split_name][1])
        
        y_true_bin = label_binarize(y_true, classes=np.arange(num_of_classes))
        
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in [3, 2, 1, 0] if num_of_classes != 2 else [0]:
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            ax.plot(fpr[i], tpr[i], color=colors[i], lw=3,
                    label=f'{class_labels_legends[i]} (AUC = {roc_auc[i]:.2f})')
            
        ax.plot([0, 1], [0, 1], 'k--', lw=2)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=18)
        ax.set_ylabel('True Positive Rate', fontsize=18)
        ax.tick_params(axis='both', labelsize=16)

        split_name = 'Validation' if split_name.lower() == 'val' else split_name
        ax.set_title(f'{split_name.capitalize()} ROC Curve', fontsize=20)
        ax.legend(loc='lower right', fontsize=14)

    axes[1].set_ylabel('')
    axes[2].set_ylabel('')

    plt.tight_layout()
    plt.savefig(f'pics/{fig_name}.svg', dpi = 300)
    plt.show()
    

def plot_confusion_matrix(num_of_classes, label_dict, extended = False, fig_name = None):
    """Plots the confusion matrices

    Args:
        num_of_classes (int): the number of classes
        label_dict (type): the labels for all three datasets, both predicted and ground true
        extended (bool, optional): if it should used the extended diagnosis. Defaults to False.
        fig_name (_type_, optional): the save name of the figure. Defaults to None.
    """
    split_names = ['train', 'val', 'test']

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

    
    for ax, split_name in zip(axes, split_names):
        class_labels_legends_extended = ['TD', 'ASD+ADHD', 'ASD', 'ADHD', 'TD+Other', 'ASD+Other', 'ADHD+Other']

        if num_of_classes == 4:
            class_labels_legends = ['TD', 'ASD+ADHD', 'ASD', 'ADHD']
            temp_label = {'TD':0, 'ASD+ADHD':1, 'ASD':2, 'ADHD':3, 'TD+Other':4, 'ASD+Other': 5, 'ADHD+Other': 6}
        else: 
            class_labels_legends = ['TD', 'Non-TD']
            temp_label = {'TD':0, 'ASD+ADHD':1, 'ASD':2, 'ADHD':3, 'TD+Other':4, 'ASD+Other': 5, 'ADHD+Other': 6}

        if extended:
            print(label_dict[split_name][0])
            label_dict[split_name][0] = [temp_label[i.replace('-', '+')] for i in label_dict[split_name][0]] 

        cf = confusion_matrix(y_true = label_dict[split_name][0], 
                              y_pred = label_dict[split_name][1])
        
        #print(label_dict[split_name][0])
        
        if extended:
            if split_name != 'train':
                class_labels_legends_extended_temp = class_labels_legends_extended
                class_labels_legends_extended_temp.remove('TD+Other')
            else:
                class_labels_legends_extended_temp = class_labels_legends_extended

            sns.heatmap(cf[:,:num_of_classes], 
                        yticklabels=class_labels_legends_extended_temp,
                        xticklabels = class_labels_legends,
                        annot=True,
                        fmt='d',
                        ax = ax,
                        cmap='Blues',
                        annot_kws={'size': 18},
                        cbar=False)
            split_name = 'Validation' if split_name.lower() == 'val' else split_name
            ax.set_title(split_name.capitalize(), fontsize=20)
            # Increase tick label font size
            ax.tick_params(axis='x', labelsize=16)
            ax.tick_params(axis='y', labelsize=16)


        else:
            sns.heatmap(cf, 
                        yticklabels= class_labels_legends,
                        xticklabels = class_labels_legends,
                        annot=True,
                        fmt='d',
                        ax = ax,
                        cmap='Blues',
                        annot_kws={'size': 18},
                        cbar=False)
            split_name = 'Validation' if split_name.lower() == 'val' else split_name
            ax.set_title(split_name.capitalize(), fontsize=20)
            ax.tick_params(axis='x', labelsize=16)
            ax.tick_params(axis='y', labelsize=16)
    
    plt.tight_layout()
    if not extended:
        plt.savefig(f'pics/{fig_name}.svg', dpi = 300)
    plt.show()

def get_parameters(yaml_file:str):
    """Maps the dict (yaml_file) to the right names

    Args:
        yaml_file (str): path to the yaml file with model metadata

    Returns:
        model specifications (dict): dict with the model specifications
    """
    with open(yaml_file, 'r') as file:
        parameters = yaml.safe_load(file)

    #align parameters
    if parameters.get('feature_names', None) == None:
        parameters['feature_names'] = parameters['feature_set']
    
    if parameters.get('edge_names', None) == None:
        parameters['edge_names'] = parameters['edge_feature_set']
    
    if parameters.get('edge_w_thres', None) == None:
        parameters['edge_w_thres'] = {'value': None} 
    
    if parameters.get('edge_w_abs', None) == None:
        parameters['edge_w_abs'] = {'value': False}

    if parameters.get('relative_edge_thres', None) == None:
        parameters['relative_edge_thres'] = {'value': None}
    
    if parameters.get('loss_name', None) == None:
        parameters['loss_name']= parameters['loss_func']
    
    return parameters

def get_data(parameters, drop_strategy, gat, dataset):
    """Loads the data for the model

    Args:
        parameters (dict): the model config parameters
        drop_strategy (int, None): the drop strategy used
        gat (bool): if model is a GAT or not
        dataset (str): if train, test or val

    Returns:
        list (object): list of graph objects
    """
    data = load_dataset(dataset = dataset, 
                            num_of_classes = parameters['num_of_classes']['value'],
                            feature_names = parameters['feature_names']['value'],
                            edge_names = parameters['edge_names']['value'],
                            edge_w_thres = parameters['edge_w_thres']['value'],
                            drop_strategy = drop_strategy,
                            edge_w_abs = parameters['edge_w_abs']['value'],
                            edge_w_relative = parameters['edge_weight_thres']['value'] if parameters['relative_edge_thres']['value'] else None,
                            GAT = gat
                                )
    return data

def evaluate_model(yaml_file:str, model_file:str, drop_strategy:int = None, gat:bool = False):
    """This function that evaluates the models. 
       It loads the data sets, and model parameters, followed 
       by plotting the roc curves and the confusion matrices

    Args:
        yaml_file (str): the path to model parameters from the yaml files
        model_file (str): the path to model weights
        drop_strategy (int, optional): if use drop strategy. Defaults to None.
        gat (bool, optional): if GAT. Defaults to False.

    Returns:
        softmax data: softmax data for each data set
    """
    parameters = get_parameters(yaml_file = yaml_file)

    train_data = get_data(parameters, drop_strategy, gat, 'train')
    val_data = get_data(parameters, drop_strategy, gat, 'val')
    test_data = get_data(parameters, drop_strategy, gat, 'test')

    train_loader = DataLoader(train_data, 
                            batch_size=parameters['batch_size']['value'])
                                
    val_loader = DataLoader(val_data, 
                            batch_size=parameters['batch_size']['value'])

    test_loader = DataLoader(test_data, 
                            batch_size=parameters['batch_size']['value'])
    
    if parameters['num_of_classes']['value'] == 2:
        num_of_classes = 1
    else:
        num_of_classes = parameters['num_of_classes']['value']

    if gat:
        model = GATFLAT(in_ = parameters['num_of_features']['value'], 
                        out_ = num_of_classes, 
                        layer_1_out = parameters['hidden_channels_1']['value'], 
                        dropout_rate = parameters['dropout']['value'], 
                        activation_ = parameters['activation']['value'], 
                        heads_ = parameters['att_heads']['value'], 
                        norm_ = parameters['layer_norm']['value'], 
                        out_func_ = parameters['loss_name']['value'], 
                        random_seed = parameters['random_seed']['value'])
    else:
        model = GCN(in_ = parameters['num_of_features']['value'], 
                        out_ = num_of_classes, 
                        layer_1_out = parameters['hidden_channels_1']['value'], 
                        dropout_rate = parameters['dropout']['value'], 
                        activation_ = parameters['activation']['value'], 
                        pool_ = parameters['pool']['value'], 
                        norm_ = parameters['layer_norm']['value'], 
                        out_func_ = parameters['loss_name']['value'], 
                        random_seed = parameters['random_seed']['value'])

    model.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
    model.eval()

    loss_func = get_loss_function(parameters['loss_name']['value'])

    cf_data = {}
    cf_ex_data = {}
    roc_curve_data = {}
    train_softmax_data = None
    for n, d in [('train', train_loader),('val', val_loader), ('test', test_loader)]:
        l, f1, acc, y_score, y, y_hat, extended_dia, softmax_data = make_interference(data = d,
                                                                                model = model, 
                                                                                loss_name = parameters['loss_name']['value'], 
                                                                                loss_func = loss_func,
                                                                                batch_size = parameters['batch_size']['value'])
        print(f"{n} loss: {l}, accuracy: {round(acc*100, 2)}, f1:{round(f1, 2)}")
        roc_curve_data[n] = [y, y_score]
        cf_data[n] = [y, y_hat]
        cf_ex_data[n] = [extended_dia, y_hat]

        if n == 'train':
            train_softmax_data = softmax_data
    plot_name = yaml_file
    plot_name = plot_name.split('/')[1].split('.')[0]
    
    plot_name = plot_name.split('_')[0] + '_' +plot_name.split('_')[1].lower()

    plot_ruc(roc_curve_data, num_of_classes = parameters['num_of_classes']['value'], fig_name = f"{plot_name}_roc")
    plot_confusion_matrix(label_dict = cf_data, num_of_classes = parameters['num_of_classes']['value'], fig_name = f"{plot_name}_confusion_matrix")
    plot_confusion_matrix(label_dict = cf_ex_data, num_of_classes = parameters['num_of_classes']['value'], extended = True)

    return train_softmax_data

def make_pca(data, info):
    """Creates a PCA, and returns the PCA data

    Args:
        data (_type_): the data to PCA
        info (_type_): the dataset

    Returns:
        _type_: the PCA transformed data
    """
    data['label'] = data['label'].replace({'ADHD-Other': 'ADHD', 
                                            'ASD-Other': 'ASD'})
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(data.drop('label', axis = 1))
    pca_data = pd.DataFrame(pca_data, columns = ['PCA_1', 'PCA_2'])
    pca_data['label'] = data['label'].tolist()
    print(f"********* {info} *********")
    print('Number of features:', len(data.columns)-1)
    print('Explained variance ratio:', pca.explained_variance_ratio_)
    return pca_data

def get_pca_plots(yaml_file:str, model_file:str, drop_strategy:int = None, gat:bool = False, dataset:str = None):
    """Creates the PCA plot for the input data and the model represntation (only GAT)

    Args:
        yaml_file (str): the path to model specifications
        model_file (str): the path to model weights
        drop_strategy (int, optional): the strategy used for dropping edges. Defaults to None.
        gat (bool, optional): if GAT or not. Defaults to False.
        dataset (str, optional): the data set to PCA. Defaults to None.

    Returns:
        DataFrame: the dataframes with the PCA data
    """
    parameters = get_parameters(yaml_file = yaml_file)

    data = get_data(parameters, drop_strategy, gat, dataset)

    loader = DataLoader(data, 
                        batch_size=parameters['batch_size']['value'])

    if parameters['num_of_classes']['value'] == 2:
        num_of_classes = 1
    else:
        num_of_classes = parameters['num_of_classes']['value']

    if gat:
        model = GATFLAT_PCA(in_ = parameters['num_of_features']['value'], 
                            out_ = num_of_classes, 
                            layer_1_out = parameters['hidden_channels_1']['value'], 
                            dropout_rate = parameters['dropout']['value'], 
                            activation_ = parameters['activation']['value'], 
                            heads_ = parameters['att_heads']['value'], 
                            norm_ = parameters['layer_norm']['value'], 
                            out_func_ = parameters['loss_name']['value'], 
                            random_seed = parameters['random_seed']['value'])

    model.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
    model.eval()
 
    before_model = []
    after_model = []

    for batch in loader:
        x, edge_index, edge_weight, batch_info = batch.x, batch.edge_index, batch.edge_attr, batch.batch
        
        #Get features before the model
        x_dense, mask = to_dense_batch(x, batch_info)
        x_dense = x_dense.reshape(x_dense.shape[0], -1)
        
        numpy_array = x_dense.detach().numpy()
        df = pd.DataFrame(numpy_array)
        df['label'] = batch.extended_dia
        before_model.append(df)
        
        #after the model
        out = model.get_pca(x, edge_index, edge_weight, batch_info)
        numpy_array = out.detach().numpy()
        df = pd.DataFrame(numpy_array)
        df['label'] = batch.extended_dia
        after_model.append(df)
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(11, 5))

    before_pca = make_pca(pd.concat(before_model), "BEFORE")
    if num_of_classes == 1:
        before_pca['label'] = before_pca['label'].apply(lambda x: 'TD' if 'TD' in x else 'Non-TD')
        hue_order = ['Non-TD', 'TD']
        palette=["purple", "#C66526"]
    else:
        hue_order = ['ADHD', 'ASD', 'ASD+ADHD', 'TD']
    
    sns.scatterplot(data=before_pca, 
                    x="PCA_1", 
                    y="PCA_2", 
                    hue = 'label',
                    ax = axes[0],
                    palette = 'colorblind' if num_of_classes != 1 else palette,
                    hue_order = hue_order,
                    legend = False)
    
    axes[0].set_title('Before', fontsize = 15)

    after_pca = make_pca(pd.concat(after_model), "AFTER")
    if num_of_classes == 1:
        after_pca['label'] = after_pca['label'].apply(lambda x: 'TD' if 'TD' in x else 'Non-TD')
    
    sns.scatterplot(data=after_pca, 
                    x="PCA_1", 
                    y="PCA_2", 
                    hue = 'label',
                    ax = axes[1],
                    palette = 'colorblind' if num_of_classes != 1 else palette,
                    hue_order = hue_order)

    axes[1].legend(title="Diagnosis", title_fontsize=13)
    axes[1].set_title('After', fontsize = 15)

    for ax in axes.flatten():
         ax.set_xlabel(ax.get_xlabel(), fontsize=14)
         ax.set_ylabel(ax.get_ylabel(), fontsize=14)
         ax.tick_params(axis='both', labelsize=13)        

    plot_name = yaml_file
    plot_name = plot_name.split('/')[1].split('.')[0]
    plot_name = plot_name.split('_')[0] + '_' +plot_name.split('_')[1].lower()

    plt.tight_layout()
    plt.savefig(f'pics/{plot_name}_pca.svg', dpi = 300)
    plt.show()

    return pd.concat(before_model), pd.concat(after_model)

def epistemic(yaml_file:str, model_file:str, dropout:float, 
              forward_passes:int, drop_strategy:int = None, gat:bool = False,
              data_set:str = None):
    """This function loads a special model that uses dropout while during interference.

    Args:
        yaml_file (str): the path to model specifications
        model_file (str): the path to model weights
        dropout (float): the dropout rate
        forward_passes (int): how many "sudo" models to run
        drop_strategy (int, optional): the drop strategy used. Defaults to None.
        gat (bool, optional): if GAT or not. Defaults to False.
        data_set (str, optional): the dataset to investigate. Defaults to None.

    Returns:
        _type_: data with the predictions (softmax)
    """

    parameters = get_parameters(yaml_file = yaml_file)

    data = get_data(parameters, drop_strategy, gat, data_set)

    loader = DataLoader(data, 
                              batch_size= 1)
    
    if parameters['num_of_classes']['value'] == 2:
        num_of_classes = 1
    else:
        num_of_classes = parameters['num_of_classes']['value']
    
    if num_of_classes == 4:
        class_labels_legends = {0:'TD', 1:'ASD+ADHD', 2:'ASD', 3:'ADHD'}
        class_index = {'TD': 0, 'ASDADHD':1, 'ASD':2, 'ADHD':3}
    else: 
        class_labels_legends = {0:'TD', 1:'Not-TD'}
        class_index = {'TD':0, 'Not-TD':1}

    model = GATFLAT_MC_DROPOUT(in_ = parameters['num_of_features']['value'], 
                        out_ = num_of_classes, 
                        layer_1_out = parameters['hidden_channels_1']['value'], 
                        dropout_rate = dropout, 
                        activation_ = parameters['activation']['value'], 
                        heads_ = parameters['att_heads']['value'], 
                        norm_ = parameters['layer_norm']['value'], 
                        out_func_ = parameters['loss_name']['value'], 
                        random_seed = parameters['random_seed']['value'])
    
    model.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))


    prediction_data = []

    #For each participant
    for batch in loader:
        x, edge_index, edge_weight, node_index = batch.x, batch.edge_index, batch.edge_attr, batch.batch
        outputs = []
        labels = []
        #For each forward pass
        for f in range(forward_passes):
            out = model.forward(x, edge_index, edge_weight, node_index) 
            outputs.append(out.unsqueeze(0))

            #Get predicted label
            if parameters['loss_name']['value'] == 'BCE':
                pre_ = out.round().tolist()[0][0]
                labels.append(class_labels_legends[pre_])
            else:
                pre_ = torch.argmax(out, dim=-1).tolist()[0]
                labels.append(class_labels_legends[pre_])
            
    
        outputs = torch.cat(outputs, dim=0)
        if parameters['loss_name']['value'] != 'BCE':
            #Log softmax -> softmax
            outputs = outputs.exp()

        mean_output = outputs.mean(dim=0)
        variance_output = outputs.var(dim=0)

        prediction_data.append({'Label': class_labels_legends[batch.y.tolist()[0]],
                                'Predicted': max(set(labels), key=labels.count),
                                'Predicted_count': dict(Counter(labels)),
                                'Mean': mean_output.tolist()[0], 
                                'Variance': variance_output.tolist()[0]})

    return pd.DataFrame(prediction_data)