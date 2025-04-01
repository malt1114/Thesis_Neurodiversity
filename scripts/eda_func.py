import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import re


def exstract_subject_information(data:pd.DataFrame) -> pd.DataFrame:
    """This function exstract information about the subject
       from the file name and adds it as columns to the dataframe

    Args:
        data (pd.DataFrame): the data with the 'subject' column

    Returns:
        pd.DataFrame: The new dataframe with the new columns
    """
    num_of_rois = data.shape[1]-1
    rois = [f"ROI_{i+1}" for i in range(num_of_rois)]
    #Get subject ID
    data['Subject_ID'] = data['subject'].apply(lambda x: re.search("\d{7}", x)[0])
    #Get session
    data['Session'] = data['subject'].apply(lambda x: re.search("session_\d", x)[0].split('_')[1])
    #Get "rest" the part of the session
    data['Rest'] = data['subject'].apply(lambda x: re.search("rest_\d", x)[0].split('_')[1])

    return data[['Subject_ID', 'Session', 'Rest'] + rois]

def get_mean_and_var_stats(data_folder:str, mac:bool = True) -> pd.DataFrame:
    """This function calculates the mean and variance
       of the ROIs pr. subject and saves them as a CSV file.

    Args:
        data_folder (str): The folder where the data is located (data/clean/XXXXXXX)
        mac (bool, optional): If the path is on a mac. Defaults to True.

    Returns:
        pd.DataFrame: the dataframes one for means and one for variance
    """
    path_addon = '.nosync' if mac else ''
    clean_path = f'data{path_addon}/clean/{data_folder}'
    file_list = os.listdir(clean_path)
    if '.DS_Store' in file_list:
        file_list.remove('.DS_Store')

    mean_pr_roi = []
    var_pr_roi = []

    for subject in tqdm(file_list): #For each subject
        subject_means = {}
        subject_variance = {}
        
        #Create subject key
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
        mean_pr_roi.append(subject_means)
        var_pr_roi.append(subject_variance)
    

    #make into dataframe, exstract subject info and save stats
    mean_pr_roi = pd.DataFrame(mean_pr_roi)
    mean_pr_roi = exstract_subject_information(data=mean_pr_roi)
    mean_pr_roi.to_csv(f'data{path_addon}/stats/{data_folder}_mean_subject.csv', sep= ";")
    
    #make into dataframe, exstract subject info and save stats
    var_pr_roi = pd.DataFrame(var_pr_roi)
    var_pr_roi = exstract_subject_information(data=var_pr_roi)
    var_pr_roi.to_csv(f'data{path_addon}/stats/{data_folder}_variance_subject.csv', sep= ";")

    return mean_pr_roi, var_pr_roi