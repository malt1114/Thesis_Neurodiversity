import numpy as np
import pandas as pd
from tqdm import tqdm
import os

def get_mean_and_var_stats(data_folder:str, mac:bool = True) -> pd.DataFrame:
    """This function calculates the mean and variance
       of the ROIs pr. subject

    Args:
        data_folder (str): _description_
        mac (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    path_addon = '.nosync' if mac else ''
    clean_path = f'data{path_addon}/clean/{data_folder}'
    file_list = os.listdir(clean_path)
    if '.DS_Store' in file_list:
        file_list.remove('.DS_Store')

    means_pr_roi = []
    var_pr_roi = []

    for subject in tqdm(file_list): #For each subject
        subject_means = {}
        subject_variance = {}

        subject_means['subject'] = subject
        subject_variance['subject'] = subject
    
        subject = np.load(f'{clean_path}/{subject}', allow_pickle = True)
        for i in subject.files: #For each ROI 
            roi = subject[i]
            values = []
            #Get all values from each timestep
            for t in range(roi.shape[0]):
                values += roi[t].tolist() #appends
            #Calculate mean and variance           
            subject_means[i] = sum(values)/len(values)
            subject_variance[i] = np.var(values)
        #Append to list
        means_pr_roi.append(subject_means)
        var_pr_roi.append(subject_variance)
    
    #save stats
    means_pr_roi = pd.DataFrame(means_pr_roi)
    means_pr_roi.to_csv(f'data{path_addon}/stats/{data_folder}_mean_subject.csv', sep= ";")

    var_pr_roi = pd.DataFrame(var_pr_roi)
    var_pr_roi.to_csv(f'data{path_addon}/stats/{data_folder}_variance_subject.csv', sep= ";")

    return means_pr_roi, var_pr_roi