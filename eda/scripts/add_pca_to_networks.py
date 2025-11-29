import os
import networkx as nx
import pandas as pd
from sklearn.decomposition import PCA

def make_pca_components(train_node_data, var_cols, mean_cols, all_cols):
    pca_components = {} #key = node, value = {'pca_var':PCA() ect.}

    #Get all the PCA components
    for n in train_node_data.node.unique():
        temp_node_data = train_node_data[train_node_data['node']== n]
        pca_var = PCA(n_components=2, svd_solver='auto').fit(temp_node_data[var_cols])
        pca_mean = PCA(n_components=2, svd_solver='auto').fit(temp_node_data[mean_cols])
        pca_all = PCA(n_components=2, svd_solver='auto').fit(temp_node_data[all_cols])
        pca_components[n] = {'pca_var': pca_var,
                            'pca_mean': pca_mean,
                            'pca_all': pca_all}
    return pca_components

def transform_node_features(network_list, pca_dict, var_cols, mean_cols, all_cols):
    networks = network_list.copy()

    for G in networks:
        G = G[1]
        for i in G.nodes.data():
            node_id = i[0]
            feature_dict = i[1].copy()

            #Get variance values, transform and add
            var_features = pd.DataFrame([{key:value for key, value in feature_dict.items() if 'var_bin' in key}],
                                        columns= var_cols)
            transform_var = pca_dict[node_id]['pca_var'].transform(var_features)[0]
            G.nodes[node_id]["pca_var_1"] = transform_var[0]
            G.nodes[node_id]["pca_var_2"] = transform_var[1]
            
            
            #Get mean values, transform and add
            mean_features = pd.DataFrame([{key:value for key, value in feature_dict.items() if 'mean_bin' in key}],
                                            columns= mean_cols)
            transform_mean = pca_dict[node_id]['pca_mean'].transform(mean_features)[0]
            G.nodes[node_id]["pca_mean_1"] = transform_mean[0]
            G.nodes[node_id]["pca_mean_2"] = transform_mean[1]
            
            #Get all values, transform and add
            all_features = pd.DataFrame([{key:value for key, value in feature_dict.items() if 'mean_bin' in key 
                                                                                            or 'var_bin' in key}],
                                            columns= all_cols)

            transform_all = pca_dict[node_id]['pca_all'].transform(all_features)[0]
            G.nodes[node_id]["pca_all_1"] = transform_all[0]
            G.nodes[node_id]["pca_all_2"] = transform_all[1]

    return networks

def get_networks(dataset):
    file_list = pd.read_csv(f'data.nosync/networks_multi/{dataset}_set_files.csv')['file'].to_list()
    networks = []
    for g in file_list:
        networks.append([g, nx.read_gml(g)])
    return networks

if __name__ =="__main__":
    os.chdir("../..")
    print(os.getcwd())
    #get train networks
    train_networks = get_networks(dataset = 'train')

    #Get the node features of the train data
    train_node_features = []
    for G in train_networks:
        G = G[1]
        for i in G.nodes.data():
            node_fea = i[1].copy()
            node_fea['node'] = i[0]
            train_node_features.append(node_fea)
    train_node_data = pd.DataFrame(train_node_features)

    #get the columns
    var_cols = [i for i in train_node_data.columns if 'var_bin' in i]
    mean_cols = [i for i in train_node_data.columns if 'mean_bin' in i]
    all_cols = var_cols + mean_cols

    pca_components = make_pca_components(train_node_data = train_node_data, 
                                        var_cols = var_cols, 
                                        mean_cols = mean_cols, 
                                        all_cols = all_cols)

    #transform train data
    train_networks = transform_node_features(network_list = train_networks, 
                                            pca_dict = pca_components, 
                                            var_cols = var_cols, 
                                            mean_cols = mean_cols, 
                                            all_cols = all_cols)
    for i in train_networks:
        nx.write_gml(i[1], i[0])

    #get validation networks
    val_networks = get_networks(dataset = 'val')
    #transform validation data
    val_networks = transform_node_features(network_list = val_networks, 
                                            pca_dict = pca_components, 
                                            var_cols = var_cols, 
                                            mean_cols = mean_cols, 
                                            all_cols = all_cols)
    for i in val_networks:
        nx.write_gml(i[1], i[0])

    #get validation networks
    test_networks = get_networks(dataset = 'test')
    #transform validation data
    test_networks = transform_node_features(network_list = test_networks, 
                                            pca_dict = pca_components, 
                                            var_cols = var_cols, 
                                            mean_cols = mean_cols, 
                                            all_cols = all_cols)
    for i in test_networks:
        nx.write_gml(i[1], i[0])
 