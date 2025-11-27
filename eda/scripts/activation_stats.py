import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

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

        if len(all_voxels) == 0:
            all_voxels = np.array([-10**6])

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
        data.to_csv(f"../data{path_addon}/stats/{data_folder}/{sub[:-7]}.csv", sep=";")
    
    pd.DataFrame(thresholds).to_csv(f"../data{path_addon}/stats/{data_folder}/{data_folder}_activation_thresholds.csv", sep=";")

    return None

def get_num_of_voxels_stats(num_of_roi:str, path_addon:str):
    data_files = pd.read_csv(f'../data{path_addon}/phenotypic/subjects_with_meta_{num_of_roi}.csv', index_col = 'Unnamed: 0')
    data_files['file_path'] = data_files['file_path'].apply(lambda x: '../'+x)

    roi_voxels = []
    for index, row in data_files.iterrows():
        sub = np.load(row['file_path'], allow_pickle = True)
        sub_vox = {'Sub ID': row['Sub ID'], 
                   'Dataset': row['Dataset']}
        for key, value in sub.items():
            sub_vox[key] = value.shape[1]
        roi_voxels.append(sub_vox)

    roi_voxels = pd.DataFrame(roi_voxels)
    roi_voxels.to_csv(f'../data{path_addon}/stats/num_of_voxels_pr_timestep_{num_of_roi}.csv')

if __name__ =="__main__":
    datasets = ['ABIDEI_7', 'ABIDEI_17', 
                'ABIDEII_7', 'ABIDEII_17', 
                'ADHD200_7', 'ADHD200_17']

    with Parallel(n_jobs=6, verbose=-1) as parallel:
            #Prepare the jobs
            delayed_funcs = [delayed(lambda x:get_activations(data_folder = x, 
                                                              mac = False))(dataset) for dataset in datasets]
            #Runs the jobs in parallel
            parallel(delayed_funcs)
    get_num_of_voxels_stats(num_of_roi = '7', path_addon = "")
    get_num_of_voxels_stats(num_of_roi = '17', path_addon = "")