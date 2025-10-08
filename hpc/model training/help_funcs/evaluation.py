import yaml
import torch
import numpy as np
import pandas as pd
import seaborn as sns
from collections import Counter
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc, f1_score, confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import PCA
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch

from models.GCN import GCN
from models.GAT import GATFLAT, GATFLAT_MC_DROPOUT, GATFLAT_PCA

from help_funcs.data_func import load_dataset
from help_funcs.param import get_loss_function

def make_interference(data, model, loss_name, loss_func, batch_size):
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
        class_labels = {0:'TD', 1:'ASD-ADHD', 2:'ASD', 3:'ADHD'}
    else: 
        class_labels = {0:'TD', 1:'Non-TD'}

    softmax_data = pd.DataFrame()
    softmax_data['label'] = all_labels
    softmax_data['label'] = softmax_data['label'].replace(class_labels)
    softmax_data['predicted'] = all_predictions
    softmax_data['predicted'] = softmax_data['predicted'].replace(class_labels)
    softmax_data['softmax_values'] = y_score
    
    if len(class_labels) != 2:
        for key, value in class_labels.items():
                softmax_data[value] = softmax_data['softmax_values'].apply(lambda x: x[key])

    return [val_loss/len(data), f1_score(all_labels, 
            all_predictions, average='micro'), y_score, 
            all_labels, all_predictions, extended_dia,
            softmax_data]

def plot_ruc(roc_curve_data, num_of_classes):
    # Define class labels and split names
    split_names = ['train', 'val', 'test']
    colors = {0:'red', 1:'green', 2:'orange', 3:'blue'}

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))
    if num_of_classes == 4:
        class_labels_legends = {0:'TD', 1:'ASD-ADHD', 2:'ASD', 3:'ADHD'}
    else: 
        class_labels_legends = {0:'TD', 1:'Non-TD'}

    for ax, split_name in zip(axes, split_names):
        y_true = np.array(roc_curve_data[split_name][0])
        y_probs = np.array(roc_curve_data[split_name][1])
        
        y_true_bin = label_binarize(y_true, classes=np.arange(num_of_classes))
        
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(num_of_classes if num_of_classes != 2 else 1):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            ax.plot(fpr[i], tpr[i], color=colors[i], lw=2,
                    label=f'{class_labels_legends[i]} (AUC = {roc_auc[i]:.2f})')

        ax.plot([0, 1], [0, 1], 'k--', lw=1)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'{split_name.capitalize()} ROC Curve')
        ax.legend(loc='lower right')
        ax.grid(True)

    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(num_of_classes, label_dict, extended = False):
    split_names = ['train', 'val', 'test']
    colors = {0:'red', 1:'green', 2:'orange', 3:'blue'}

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))


    for ax, split_name in zip(axes, split_names):
        if num_of_classes == 4:
            class_labels_legends = ['TD', 'ASD-ADHD', 'ASD', 'ADHD']
            class_labels_legends_extended = ['TD', 'ASD-ADHD', 'ASD', 'ADHD', 'TD-Other', 'ASD-Other', 'ADHD-Other']
            temp_label = {'TD':0, 'ASD-ADHD':1, 'ASD':2, 'ADHD':3, 'TD-Other':4, 'ASD-Other': 5, 'ADHD-Other': 6}
        else: 
            class_labels_legends = ['TD', 'Non-TD']
            class_labels_legends_extended = ['TD', 'Non-TD', 'TD-Other', 'ASD-Other', 'ADHD-Other']
            temp_label = {'TD':0, 'ASD-ADHD':1, 'ASD':1, 'ADHD':1, 'TD-Other':2, 'ASD-Other': 3, 'ADHD-Other': 4}

        if extended:
            label_dict[split_name][0] = [temp_label[i] for i in label_dict[split_name][0]] 

        cf = confusion_matrix(y_true = label_dict[split_name][0], 
                              y_pred = label_dict[split_name][1])

        if extended:
            if split_name != 'train':
                class_labels_legends_extended_temp = class_labels_legends_extended
                class_labels_legends_extended_temp.remove('TD-Other')
            else:
                class_labels_legends_extended_temp = class_labels_legends_extended

            sns.heatmap(cf[:,:num_of_classes], 
                        yticklabels=class_labels_legends_extended_temp,
                        xticklabels = class_labels_legends_extended_temp[:num_of_classes],
                        annot=True,
                        fmt='d',
                        ax = ax,
                        cmap='Blues')
        else:
            sns.heatmap(cf, 
                        yticklabels=class_labels_legends,
                        xticklabels = class_labels_legends,
                        annot=True,
                        fmt='d',
                        ax = ax,
                        cmap='Blues')
    
    plt.tight_layout()
    plt.show()

def get_parameters(yaml_file:str):
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
        l, f1, y_score, y, y_hat, extended_dia, softmax_data = make_interference(data = d,
                                                                                model = model, 
                                                                                loss_name = parameters['loss_name']['value'], 
                                                                                loss_func = loss_func,
                                                                                batch_size = parameters['batch_size']['value'])
        print(f"{n} loss: {l} f1:{f1}")
        roc_curve_data[n] = [y, y_score]
        cf_data[n] = [y, y_hat]
        cf_ex_data[n] = [extended_dia, y_hat]

        if n == 'train':
            train_softmax_data = softmax_data
    
    plot_ruc(roc_curve_data, num_of_classes = parameters['num_of_classes']['value'])
    plot_confusion_matrix(label_dict = cf_data, num_of_classes = parameters['num_of_classes']['value'])
    plot_confusion_matrix(label_dict = cf_ex_data, num_of_classes = parameters['num_of_classes']['value'], extended = True)

    return train_softmax_data

def make_pca(data, info):
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
    else:
        hue_order = ['ADHD', 'ASD', 'ASD-ADHD', 'TD']
    sns.scatterplot(data=before_pca, 
                    x="PCA_1", 
                    y="PCA_2", 
                    hue = 'label',
                    ax = axes[0],
                    palette = 'colorblind',
                    hue_order = hue_order,
                    legend = False)
    axes[0].set_title('Before')

    after_pca = make_pca(pd.concat(after_model), "AFTER")
    if num_of_classes == 1:
        after_pca['label'] = after_pca['label'].apply(lambda x: 'TD' if 'TD' in x else 'Non-TD')
    sns.scatterplot(data=after_pca, 
                    x="PCA_1", 
                    y="PCA_2", 
                    hue = 'label',
                    ax = axes[1],
                    palette = 'colorblind',
                    hue_order = hue_order)
    axes[1].set_title('After')

    plt.tight_layout()
    plt.show()


    return pd.concat(before_model), pd.concat(after_model)


def epistemic(yaml_file:str, model_file:str, dropout:float, 
              forward_passes:int, drop_strategy:int = None, gat:bool = False,
              data_set:str = None):

    parameters = get_parameters(yaml_file = yaml_file)

    data = get_data(parameters, drop_strategy, gat, data_set)

    loader = DataLoader(data, 
                              batch_size= 1)
    
    if parameters['num_of_classes']['value'] == 2:
        num_of_classes = 1
    else:
        num_of_classes = parameters['num_of_classes']['value']
    
    if num_of_classes == 4:
        class_labels_legends = {0:'TD', 1:'ASD-ADHD', 2:'ASD', 3:'ADHD'}
        class_index = {'TD': 0, 'ASDADHD':1, 'ASD':2, 'ADHD':3}
    else: 
        class_labels_legends = {0:'TD', 1:'Non-TD'}
        class_index = {'TD':0, 'Non-TD':1}

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