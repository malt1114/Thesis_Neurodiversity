import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict
from nilearn import image as nimg
from joblib import Parallel, delayed

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=RuntimeWarning) 

def get_roi_dict(mask: np.ndarray, filter_mask: np.ndarray) -> Dict[int,np.ndarray]: # ndarray is type .array is function to make type
    """This function returns a dict of the ROI masks

    Returns:
        dict: key is the roi number and value is a np array of bools
    """
    if filter_mask is not None:
        filter_mask = filter_mask.get_fdata()
        filter_mask = filter_mask.astype(np.int16)
        #Filter mask should be 1 (brain)
        filter_mask = filter_mask == 1
    #Get number of unique ROIs
    num_of_roi = np.unique(mask).tolist()

    #Create dict of arrays and their ROI
    roi_dict = {}
    for i in num_of_roi[1:]: #Avoid creating one for the "non-brain" (0 values)
        roi = mask == i
        roi = np.squeeze(roi)
        if filter_mask is not None:
            roi = (roi == True) & (filter_mask == True)
        roi_dict[i] = roi
    return roi_dict

def get_image_rois(roi_dict:dict, img: np.array, standardize: bool = False) -> dict:
    """This creates a np array of (time, voxels) for each
       of the ROI in the roi dict. It then returns a 
       dictionary with each key, value pair being a 
       ROI

    Args:
        roi_dict (dict): the ROI masks
        img (np.array): The image to segment

    Returns:
        dict: key is the ROI, value is a np.array of the ROI over time
    """
    roi_arrays = {}
    all_voxels_arr = []
    
    #Make one numpy array for each of the ROIs
    for key, value in roi_dict.items():
        temp_mask = value
        roi_time = []
        #For each timestep
        for t in range(img.shape[-1]):
            roi_time.append(img[:,:,:,t][temp_mask])
        all_voxels_arr += roi_time
        roi_arrays[f"ROI_{int(key)}"] = np.array(roi_time).astype(np.float32)
    
    if standardize:
        #Calculate statistics and perform standardization
        all_voxels = np.concatenate(all_voxels_arr)
        std = all_voxels.std()
        mean = all_voxels.mean()

        for key, value in roi_arrays.items():
            roi_arrays[key] = (value-mean)/std
    
    return roi_arrays

def read_img_fit_mask(img_path:str, mask, filter_mask:str) -> np.array:
    """This function: 
        1. loads the image (scan)
        2. fits the image to the mask
        3. filters the image based on a threshold
        4. makes the mask and image into np arrays
        5. returns the numpy arrays

    Args:
        img_path (str): The path to the image/scan file
        mask (_type_): The mask used

    Returns:
        fitted_mask: np.array
        img: np.array
    """
    #Load image
    img = nimg.load_img(img_path)
    #Fit parcellation to the filter mask
    fitted_mask = nimg.resample_img(mask, 
                                    target_affine=filter_mask.affine, 
                                    interpolation='nearest',
                                    target_shape=filter_mask.shape[:3], 
                                    force_resample = True,
                                    copy_header=True) 
    
    #Make into numpy arrays
    fitted_mask = fitted_mask.get_fdata()
    img = img.get_fdata()

    return fitted_mask, img

def clean_scans(folder: str, hpc:bool, target_folder:str, mask_in, mac: bool = True) -> None:
    """This function takes a folder (str) that is places
       in data.nosync/clean, and cleans it into 
       numpy files.

    Args:
        folder (str): the folder with files to clean
    """
    #windows vs mac :)
    if mac == True:
        sync = ".nosync"
    elif mac == False:
        sync = ""

    if hpc:
        #If on the HPC, we need to iterrate over all the folders
        #to find the files that we need
        #Get all paths to the participants
        participant_folder_path = f'../data{sync}/preprocessed/{folder}/output/pipeline_cpac-nc-custom-pipeline'

        file_list = []

        missing = []
        #For each participant
        for p in os.listdir(participant_folder_path):
            sessions = os.listdir(f"{participant_folder_path}/{p}")
            #For each session
            for s in sessions:
                p_files = os.listdir(f"{participant_folder_path}/{p}/{s}/func/")
                scan_file_one = f"{participant_folder_path}/{p}/{s}/func/{p}_{s}_task-rest_run-1_space-MNI152NLin6ASym_desc-preproc_bold.nii.gz"
                scan_file_two = f"{participant_folder_path}/{p}/{s}/func/{p}_{s}_task-rest_run-2_space-MNI152NLin6ASym_desc-preproc_bold.nii.gz"

                if f"{p}_{s}_task-rest_run-1_space-MNI152NLin6ASym_desc-preproc_bold.nii.gz" in p_files:
                    #e.g participant_folder_path/sub-29177/ses-1/func/sub-29177_ses-1_task-rest_run-1_space-MNI152NLin6ASym_reg-default_desc-preproc_bold.nii.gz
                    file_list.append((scan_file_one,
                                    f"{participant_folder_path}/{p}/{s}/func/{p}_{s}_task-rest_run-1_space-MNI152NLin6ASym_res-3mm_desc-bold_mask.nii.gz"))
                else:
                    missing.append(scan_file_one)
                
                if f"{p}_{s}_task-rest_run-2_space-MNI152NLin6ASym_desc-preproc_bold.nii.gz" in p_files:
                    file_list.append((scan_file_two,
                                    f"{participant_folder_path}/{p}/{s}/func/{p}_{s}_task-rest_run-2_space-MNI152NLin6ASym_res-3mm_desc-bold_mask.nii.gz"))
                    
        #Print missing scans
        print('Missing files:', flush = True)
        for m in missing:
            print(m, flush = True)

    else:
        #Path with the fmri scans
        preprocessed_path = f'../data{sync}/preprocessed/{folder}'
        file_list = os.listdir(preprocessed_path)
        file_list = [f"{preprocessed_path}/{i}" for i in file_list]
    
    #Path where to save the cleaned data
    clean_path = f'../data{sync}/clean/{target_folder}'

    mask = nimg.load_img(f'../data{sync}/{mask_in}')

    size_data = []

    for file in tqdm(file_list):
        if type(file) != str:
            #read filter mask
            filter_mask = nimg.load_img(file[1])
            #read image and fit parcellation mask to scan mask 
            fitted_mask, img = read_img_fit_mask(img_path = file[0],
                                                mask = mask,
                                                filter_mask= filter_mask)
            #Get size of all regions
            size_data.append([file[0], fitted_mask[(fitted_mask != 0.0) & (fitted_mask != 0)].flatten().shape])

            roi_dict = get_roi_dict(mask = fitted_mask, filter_mask = filter_mask)
            roi_arrays = get_image_rois(roi_dict = roi_dict, img = img, standardize = True)
            #Save
            np.savez_compressed(f"{clean_path}/{file[0].split('/')[-1][:-7]}", **roi_arrays, allow_pickle=True)

        else:
            fitted_mask, img = read_img_fit_mask(img_path = file,
                                                mask = mask,
                                                filter_mask= None)
            #Get ROI
            roi_dict = get_roi_dict(mask = fitted_mask, filter_mask = None)
            #Get roi arrays over time
            roi_arrays = get_image_rois(roi_dict = roi_dict, img = img, standardize = False)
        
            #Save
            np.savez_compressed(f"{clean_path}/{file.split('/')[-1][:-4]}", **roi_arrays, allow_pickle=True)
    pd.DataFrame(size_data, columns=['file', '#voxels']).to_csv(f'{clean_path}/#Number of voxels {mask_in}.csv')        


if __name__ =="__main__":

    datasets = [
        ['ABIDEI', True, 'ABIDEI_7','Yeo2011_7Networks_MNI152_FreeSurferConformed1mm_LiberalMask.nii', False],
        ['ABIDEI', True, 'ABIDEI_17','Yeo2011_17Networks_MNI152_FreeSurferConformed1mm_LiberalMask.nii', False],
        #ABIDE II
        ['ABIDEII', True, 'ABIDEII_7','Yeo2011_7Networks_MNI152_FreeSurferConformed1mm_LiberalMask.nii', False],
        ['ABIDEII', True, 'ABIDEII_17','Yeo2011_17Networks_MNI152_FreeSurferConformed1mm_LiberalMask.nii', False],
        #ADHD200
        ['ADHD200', True, 'ADHD200_7','Yeo2011_7Networks_MNI152_FreeSurferConformed1mm_LiberalMask.nii', False],
        ['ADHD200', True, 'ADHD200_17','Yeo2011_17Networks_MNI152_FreeSurferConformed1mm_LiberalMask.nii', False]
        ]
    
    with Parallel(n_jobs=6, verbose=-1) as parallel:
            #Prepare the jobs
            delayed_funcs = [delayed(lambda x:clean_scans(folder = x[0], 
                                                         hpc = x[1], 
                                                         target_folder=x[2], 
                                                         mask_in = x[3], 
                                                         mac=x[4]))(dataset) for dataset in datasets]
            #Runs the jobs in parallel
            parallel(delayed_funcs)