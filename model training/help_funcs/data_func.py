#Python basic
import math
import os 

#Torch
import torch
from torch_geometric.utils import from_networkx

#Other
import networkx as nx
import pandas as pd


#Load dataset
def load_dataset(dataset:str, num_of_classes:int, feature_names:list[str], 
                 edge_names:list[str], edge_w_thres:float = None, 
                 drop_strategy:int = None, edge_w_abs:bool = False, 
                 edge_w_relative:int = None,
                 GAT:bool = False):
    """This function loads the dataset, and applies the different
       edge weight thresholds, selects node and edge features

    Args:
        dataset (str): train, val or test set
        num_of_classes (int): the number of classes
        feature_names (list[str]): the node features
        edge_names (list[str]): the edge weigths use, e.g. mean_82
        edge_w_thres (float, optional): if a threshold should be set for the edge weights. Defaults to None.
        drop_strategy (int, optional): if a drop strategy should be set. Defaults to None.
        edge_w_abs (bool, optional): if edge weights is absolute or not. Defaults to False.
        edge_w_relative (int, optional): if a relative edge weigth threshold should be used. Defaults to None.
        GAT (bool, optional): if the model is GAT or not. Defaults to False.

    Returns:
        list (objects): the graphs objects as tensors
    """

    node_feature_set = get_node_features(feature_names)
    
    if num_of_classes == 2:
        class_dict = {'TD': 0, 'ASD-ADHD':1, 'ASD':1, 'ADHD':1}
    else:
        class_dict = {'TD': 0, 'ASD-ADHD':1, 'ASD':2, 'ADHD':3}

    file_list = pd.read_csv(f'data.nosync/networks_multi{"_gat" if GAT else ""}/{dataset}_set_files.csv')['file'].to_list()
    meta_data = pd.read_csv(f'data.nosync/phenotypic/meta_data.csv')
    meta_data['Sub ID'] = meta_data['Sub ID'].apply(lambda x: str(x).zfill(7))
    meta_data['extended_dia'] = meta_data['Diagnosis'] + '-' + meta_data['Co-Diagnosis']

    data_list = []

    for i in file_list:
        network_class = i.split('/')[-1].split('_')[3]
        G = nx.read_gml(f'{i}')

        if GAT: 
            edges_to_remove = []
            
            for e in G.edges(data= True):
                u,v, data = e[0], e[1], e[2]

                features_to_remove = []
                for name, value in data.items():
                    #If not in edge names
                    if name not in edge_names:
                        features_to_remove.append(name)
                    else:
                        #If value is nan
                        if math.isnan(G[u][v][name]):
                            G[u][v][name] = 0

                        #If absolute value
                        if edge_w_abs:
                            G[u][v][name] = abs(G[u][v][name])
                        
                        #If threshold is not None
                        if edge_w_thres != None and G[u][v][name] < edge_w_thres:
                            G[u][v][name] = 0
                        
                        #If relative threshold is not non, and the absolute weight is below
                        if edge_w_relative != None and abs(G[u][v][name]) < edge_w_relative:
                            G[u][v][name] = 0

                #Remove not need features
                for f in features_to_remove:
                    del e[2][f]

                #if drop strategy is specified
                if drop_strategy != None:
                    #Count number of edge weights with 0
                    zero_count = list(G[u][v].values()).count(0)
                    #If above or equal to thres hold, remove edge
                    if zero_count >= drop_strategy:
                        edges_to_remove.append((u, v)) 
            if drop_strategy != None:
                for e in edges_to_remove:
                    G.remove_edge(e[0], e[1])

        else:
            e_to_remove = []
            for e in G.edges():
                u, v = e
                for key, value in G[u][v].items():
                    feature_name = G[u][v][key]['feature_name']
                    #If absolute values
                    if edge_w_abs:
                        G[u][v][key]['feature_value'] = abs(value['feature_value']) if not math.isnan(value['feature_value']) else 0

                    else:
                        G[u][v][key]['feature_value'] = value['feature_value'] if not math.isnan(value['feature_value']) else 0
                    #If there is set a threshold for the edge weight
                    if edge_w_thres != None:
                        #Remove edge if no weights or is not used
                        if G[u][v][key]['feature_value'] < edge_w_thres or feature_name not in edge_names:
                            e_to_remove.append((u,v,key))
                    else:
                        #Remove edge if not used
                        if feature_name not in edge_names:
                            e_to_remove.append((u,v,key))

            G.remove_edges_from(e_to_remove)
        
        participant = i.split('/')[2].split('_')

        extended_dia = meta_data[(meta_data['Sub ID'] == participant[0]) & (meta_data['Dataset'] == participant[2])]

        graph = from_networkx(G, group_node_attrs = node_feature_set, group_edge_attrs= edge_names if GAT else ['feature_value'])
        graph.y = class_dict[network_class]

        graph.extended_dia = participant[3] if str(extended_dia['extended_dia'].tolist()[0]) == 'nan' else str(extended_dia['extended_dia'].tolist()[0])
        data_list.append(graph)

    for data in data_list:
        data.x = data.x.to(torch.float32)
        if hasattr(data, 'edge_attr'):
            data.edge_attr = data.edge_attr.to(torch.float32)

    return data_list

def get_node_features(node_features:list[str]):
    """Selects the node feature names

    Args:
        node_features (list[str]): the names, e.g var and mean

    Returns:
        final_feature_set (list[str]): the names of the node features
    """
    final_feature_set = []
    if 'var_bin' in node_features:
        final_feature_set += ['var_bin_21_1', 'var_bin_21_2','var_bin_21_3', 'var_bin_21_4',
                              'var_bin_21_5', 'var_bin_21_6','var_bin_21_7', 'var_bin_21_8']
    if 'mean_bin' in node_features:
        final_feature_set += ['mean_bin_21_1', 'mean_bin_21_2','mean_bin_21_3', 'mean_bin_21_4',
                              'mean_bin_21_5', 'mean_bin_21_6','mean_bin_21_7', 'mean_bin_21_8']
    if 'pca_mean' in node_features:
        final_feature_set += ['pca_mean_1', 'pca_mean_2']
    
    if 'pca_var' in node_features:
        final_feature_set += ['pca_var_1', 'pca_var_2']

    if 'pca_all' in node_features:
        final_feature_set += ['pca_all_1', 'pca_all_2']

    if final_feature_set == []:
        print('No such feature set:', node_features)
    else:
        return final_feature_set
    

def create_gat_network(path):
    """Transforms a multi edges into a single edge

    Args:
        path (str): path to network

    Returns:
        _type_: the gat network
    """
    network = nx.read_gml(path)

    new_network = nx.Graph()

    new_edges = []
    for u,v in set(network.edges()):
        new_edge = [u,v,{}]
        for name, value in network[u][v].items():
            if value['feature_name']:
                new_edge[-1][value['feature_name']] = value['feature_value']
        new_edges.append(new_edge)

    for n in network.nodes(data=True):
        new_network.add_node(n[0], **n[1])

    new_network.add_edges_from(new_edges)
    return new_network

def transform_to_gat():
    """
    Transforms the networks from single edge to multi edge
    """
    #Transform all networks
    file_list = [f for f in os.listdir('../data.nosync/networks_multi/') if '.gml' in f ]
    for i in file_list:
        G = create_gat_network(f"../data.nosync/networks_multi/{i}")
        nx.write_gml(G, f"../data.nosync/networks_multi_gat/{i}")

    #Transfrom datasets
    for s in ['train', 'test', 'val']:
        file_data = pd.read_csv(f'../data.nosync/networks_multi/{s}_set_files.csv')
        files = file_data['file'].to_list()

        file_data['file'] = file_data['file'].apply(lambda x: x.replace('networks_multi', 'networks_multi_gat'))
        
        for i in files:
            G = create_gat_network(i)
            nx.write_gml(G, i.replace('networks_multi', 'networks_multi_gat'))

        file_data.to_csv(f'../data.nosync/networks_multi_gat/{s}_set_files.csv',index= False)