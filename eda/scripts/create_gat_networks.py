import networkx as nx
import os
import pandas as pd

def create_network(path):
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


if __name__ =="__main__":
    #Transform all networks
    file_list = [f for f in os.listdir('../../data.nosync/networks_multi/') if '.gml' in f ]
    for i in file_list:
        G = create_network(f"../../data.nosync/networks_multi/{i}")
        nx.write_gml(G, f"../../data.nosync/networks_multi_gat/{i}")

    #Transfrom datasets
    for s in ['train', 'test', 'val']:
        file_data = pd.read_csv(f'../../data.nosync/networks_multi/{s}_set_files.csv')
        files = file_data['file'].to_list()

        file_data['file'] = file_data['file'].apply(lambda x: x.replace('networks_multi', 'networks_multi_gat'))
        
        for i in files:
            G = create_network(i)
            nx.write_gml(G, i.replace('networks_multi', 'networks_multi_gat'))

        file_data.to_csv(f'../../data.nosync/networks_multi_gat/{s}_set_files.csv',index= False)
