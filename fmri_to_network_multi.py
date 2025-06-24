from tqdm import tqdm
import numpy as np
import pandas as pd
import networkx as nx
from itertools import combinations

class FmriToNetwork():

    def __init__(self, subject_id:int, dataset:str, diagnosis:str, bins: list[int], num_rois:int ,mean_data:str, hpc:bool):
        self.subject_id = self.fill_subject_id(subject_id = subject_id)
        self.dataset = dataset
        self.diagnosis = diagnosis
        self.scan_data = self.read_mean_data(path = mean_data, hpc = hpc)
        self.run = [i for i in mean_data.split('_') if 'run-' in i][0]
        self.bins = bins
        self.num_rois = num_rois
        self.roi_names = [f"ROI_{i+1}" for i in range(num_rois)]
        self.hpc = '' if hpc else '.nosync'

    def fill_subject_id(self, subject_id:int):
        """Fills the subject id, 
           with 0's to lenght of 7

        Args:
            subject_id (str): the subject id
        """
        subject_id = str(subject_id).zfill(7)
        return subject_id

    def read_mean_data(self, path:str, hpc:bool) -> pd.DataFrame:
        """Removes .nosync from the data path,
           if it is running on the hpc 

        Args:
            path (str): the data path to the mean values
            hpc (bool): if on hpc or not
        """
        if hpc == True:
            path = path.replace('.nosync', '')
        
        return np.load(path, allow_pickle = True)

    def get_bin_slices(self) -> dict:
        """This function creates the bin slices
           of the indexes, e.g [0:i] [i,i+bin] ect. 

        Returns:
            dict: key is the bin size, value is a list of intervals
        """

        bin_dict = {}
        for bin in self.bins:
            bin_slices = []
            for i in range(bin, self.scan_data['ROI_1'].shape[0]+1, bin):
                bin_slices.append([i-bin,i])
            bin_dict[bin] = bin_slices
        return bin_dict
    
    def calculate_node_bin_feature(self, bin_intervals: list, stat_type:str, as_dict:bool = False) -> dict:
        stat_node_dict = {}
        
        #For each roi
        for roi in self.roi_names:
            temp_data = self.scan_data[roi]
            roi_bin_stat = {} if as_dict else []
            bin_count = 1
            #For each bin interval 
            for interval in bin_intervals:
                interval_size = interval[1]-interval[0]
                #Flatten the time steps, for quick calculation
                bin_data = temp_data[interval[0]:interval[1]].flatten()
                #Select type of stats
                if stat_type == 'mean':
                    if as_dict:
                        roi_bin_stat[f"mean_bin_{interval_size}_{bin_count}"] = float(bin_data.mean())
                        bin_count += 1
                    else: 
                        roi_bin_stat.append(float(bin_data.mean()))
                elif stat_type == 'var':
                    if as_dict:
                        roi_bin_stat[f"var_bin_{interval_size}_{bin_count}"] = float(bin_data.var())
                        bin_count += 1
                    else: 
                        roi_bin_stat.append(float(bin_data.var()))
                else:
                    print('The stat_type of calculate_node_bin_feature, har to be in ["var", "mean"]')
            #Add stat for node to the stat node dict
            stat_node_dict[roi+f"_{stat_type}"] = roi_bin_stat

        return stat_node_dict
    
    def calculate_correlation_between_rois(self, bin_slices:dict):
        edge_pairs = sorted(map(sorted, combinations(set(self.roi_names), 2)))
        edge_corr_data = {' '.join(k):list() for k in edge_pairs}

        edges = [] #[u, v, {'feature': value}]

        #For each bin
        for bin_size, intervals in bin_slices.items():
            bin_mean_data = self.calculate_node_bin_feature(bin_intervals = intervals, stat_type= 'mean')
            bin_var_data = self.calculate_node_bin_feature(bin_intervals = intervals, stat_type= 'var')

            #For each ROI-pair calculate the correlation coef
            for pair in edge_pairs:
                corr_coef_mean = np.corrcoef(np.array([bin_mean_data[pair[0]+"_mean"], 
                                                       bin_mean_data[pair[1]+"_mean"]]))[0,1]
                corr_coef_var = np.corrcoef(np.array([bin_var_data[pair[0]+"_var"], 
                                                      bin_var_data[pair[1]+"_var"]]))[0,1]
                
                edges.append([int(pair[0].split('_')[1]), int(pair[1].split('_')[1]), {'feature_name':f'corr_mean_bin_{bin_size}', 'feature_value':float(corr_coef_mean)}])
                edges.append([int(pair[0].split('_')[1]), int(pair[1].split('_')[1]), {'feature_name':f'corr_var_bin_{bin_size}', 'feature_value':float(corr_coef_var)}])                
    
        return edges

    def make_node_feature_to_nx_format(self, feature_dict:dict, stat_type:str):
        nodes_attr_data = {}

        for key, value in feature_dict.items():
            node = key[:5]
            node_attr = {}
            for i in range(len(value)):
                node_attr[f"{stat_type}_bin_num_{i+1}"] = value[i]
            nodes_attr_data[node] = node_attr
        return nodes_attr_data

    def create_network(self):
        #Get bin slices
        bin_slices = self.get_bin_slices()
        
        #NODE FEATURES
        #Calculate mean for all of the smallest bin size 
        mean_values_bin = self.calculate_node_bin_feature(bin_intervals = bin_slices[min(self.bins)],
                                                          stat_type='mean',
                                                          as_dict = True)
        
        
        #Calculate the variance of the smallest bin size
        var_values_bin = self.calculate_node_bin_feature(bin_intervals = bin_slices[min(self.bins)], 
                                                         stat_type='var',
                                                         as_dict = True)
        
        node_attributes = {}
        for roi in self.roi_names:
            node_attributes[int(roi.split('_')[1])] = {**var_values_bin[roi+"_var"], **mean_values_bin[roi+"_mean"]}
        
        #EDGE FEATURES
        #Calculates the edge features, as the correlation between two roi's means and variance
        edges = self.calculate_correlation_between_rois(bin_slices)

        #CREATE THE NETWORK
        network = nx.MultiGraph()
        for e in edges:
            network.add_edge(e[0], e[1], **e[2])
       
        #network.add_weighted_edges_from(edges)
        nx.set_node_attributes(network, node_attributes)
        print(f"Number of nodes: {network.number_of_nodes()}, Number of edges: {network.number_of_edges()}", flush= True)

        #Save the data
        #TODO: Save network as
        nx.write_gml(network, f"data{self.hpc}/networks_multi/{self.subject_id}_{self.run}_{self.dataset}_{self.diagnosis}_{self.num_rois}.gml")

if __name__ =="__main__":
    """
    FmriToNetwork(subject_id = 'test', 
                    dataset = 'test', 
                    diagnosis = 'test', 
                    bins = [21, 42, 86], 
                    num_rois = 17,
                    mean_data = "data.nosync/clean/sub-0050952_ses-1_task-rest_run-1_space-MNI152NLin6ASym_desc-preproc_bold.npz",
                    hpc=False).create_network()
    
    """
    file_data = pd.read_csv('data/phenotypic/subjects_with_meta_17.csv', index_col= 'Unnamed: 0')
    file_data['Co-Diagnosis'] = file_data['Co-Diagnosis'].apply(lambda x: '' if str(x) == 'nan' or str(x) == 'Other' else x)
    file_data['Full Diagnosis'] = file_data['Diagnosis'] + '-' + file_data['Co-Diagnosis']
    file_data['Full Diagnosis'] = file_data['Full Diagnosis'].apply(lambda x: x.replace('-', '') if x[-1] == '-' else x)
    file_data['file_path'] = file_data['file_path'].apply(lambda x: x.replace('.nosync', ''))
    file_data = file_data[['Sub ID', 'Dataset', 'file_path', 'Full Diagnosis']]
    file_data = file_data.values.tolist()
    
    for sub in tqdm(file_data):
        FmriToNetwork(subject_id = sub[0], 
                    dataset = sub[1], 
                    diagnosis = sub[3], 
                    bins = [21, 42, 86], 
                    num_rois = 17,
                    mean_data = sub[2],
                    hpc=True).create_network()
