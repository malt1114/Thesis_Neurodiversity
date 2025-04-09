from tqdm import tqdm
import pandas as pd
import numpy as np
import os


def get_activations(data_folder:str, mac:bool = True):
    """This function calculates the mean and variance
       of the ROIs pr. subject and saves them as a CSV file.

    Args:
        data_folder (str): The folder where the data is located (data/clean/XXXXXXX)
        mac (bool, optional): If the path is on a mac. Defaults to True.

    Returns:
        pd.DataFrame: the dataframes one for means and one for variance
    """
    path_addon = '.nosync' if mac else ''
    clean_path = f'../data{path_addon}/clean/{data_folder}'
    file_list = os.listdir(clean_path)
    if '.DS_Store' in file_list:
        file_list.remove('.DS_Store')

    thresholds = []

    for sub in tqdm(file_list): #For each subject
        #To store the subject ROI means at each timestep 
        subject_roi_means_time = []
    
        all_voxels = []

        subject = np.load(f'{clean_path}/{sub}', allow_pickle = True)
        #For each ROI 
        for i in subject.files:
            roi = subject[i]
            #Add all voxes to threshold list
            all_voxels += roi.flatten().tolist()
            
            #Make dict of roi means over time
            roi_mean_time = {'ROI': i}

            #For each timestep get mean
            for t in range(roi.shape[0]):
                roi_mean_time[t+1] = float(roi[t].mean())
            
            #Add region to subject
            subject_roi_means_time.append(roi_mean_time)

        all_voxels = np.array(all_voxels)

        thresholds.append({ 'Subject': sub[:-4], 
                            '10th': np.quantile(all_voxels, 0.10),
                            '25th': np.quantile(all_voxels, 0.25),
                            '50th': np.quantile(all_voxels, 0.50),
                            '75th': np.quantile(all_voxels, 0.75),
                            '90th': np.quantile(all_voxels, 0.90)})

        #Make dataframe and save
        data = pd.DataFrame(subject_roi_means_time).T
        new_header = data.iloc[0] #grab the first row for the header
        data = data[1:] #take the data less the header row
        data.columns = new_header #set the header row as the df header
        data.to_csv(f"../data{path_addon}/stats/{data_folder}/{sub[:-4]}.csv", sep=";")
    
    pd.DataFrame(thresholds).to_csv(f"../data{path_addon}/stats/{data_folder}_activation_thresholds.csv", sep=";")

    return None

get_activations(data_folder = 'ADHD200_7')
get_activations(data_folder = 'ADHD200_17')