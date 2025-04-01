from nilearn import image as nimg
from tqdm import tqdm
import numpy as np
import os
from typing import Dict

from nilearn.image import threshold_img
from nilearn import plotting as nplot

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

def get_roi_dict(mask: np.ndarray) -> Dict[int,np.ndarray]: # ndarray is type .array is function to make type
    """This function returns a dict of the ROI masks

    Returns:
        dict: key is the roi number and value is a np array of bools
    """
    #Get number of unique ROIs
    num_of_roi = np.unique(mask).tolist()
    #print(num_of_roi)
    #Create dict of arrays and their ROI
    roi_dict = {}
    for i in num_of_roi[1:]: #Avoid creating one for the "non-brain" (0 values)
        roi = mask == i
        roi_dict[i] = roi[:,:,:,0]
    return roi_dict

def get_image_rois(roi_dict:dict, img: np.array) -> dict:
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
    
    #Make one numpy array for each of the ROIs
    for key, value in roi_dict.items():
        temp_mask = value
        roi_time = []
        #For each timestep
        for t in range(img.shape[-1]):
            roi_time.append(img[:,:,:,t][temp_mask])
        roi_arrays[f"ROI_{int(key)}"] = np.array(roi_time).astype(np.float32)
    return roi_arrays

def clean_scans(folder: str, target_folder:str, mask, mac: bool = True) -> None:
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

    preprocessed_path = f'../data{sync}/preprocessed/{folder}'
    clean_path = f'../data{sync}/clean/{target_folder}'

    mask = nimg.load_img(f'../data{sync}/{mask}')

    file_list = os.listdir(preprocessed_path)
    for file_name in tqdm(file_list):
        if file_name == file_list[0]:
            continue
        #Load image
        img = nimg.load_img(f"{preprocessed_path}/{file_name}")

        #Fit mask to image
        fitted_mask = nimg.resample_img(mask, 
                                        target_affine=img.affine, #NC: not sure here as i think we originally used one example image as basis for affine transform, 
                                        #maybe there are differences doing things this way? this is potentially slower but probably leads to the same result
                                        interpolation='nearest',
                                        target_shape=img.shape[:3], 
                                        force_resample = True, #NC: not sure, new to a recent version of the library
                                        copy_header=True) 
        
        img = threshold_img( # mask the image and threshold
            img,
            threshold= np.float32(0.0), #TODO change if not this dataset with gaussian blurring
            two_sided=True,
            copy_header=True,
            copy=True,
            mask_img= nimg.binarize_img(fitted_mask, copy_header=True)
            )
        
        # # #check visually
        # nplot.plot_roi(mask, img.slicer[:,:,:,30],cut_coords=[-4,14,7])

        # nplot.plot_roi(fitted_mask, img_t.slicer[:,:,:,30],cut_coords=[-4,14,7])
        # nplot.plot_img(img_t.slicer[:,:,:,30],threshold=0,cut_coords=[-4,14,7])

        # nplot.show()

        #Make into numpy arrays
        fitted_mask = fitted_mask.get_fdata()
        img = img.get_fdata()

        #Get ROI
        roi_dict = get_roi_dict(mask = fitted_mask) 
        #print(roi_dict.keys())
        #Get roi arrays over time
        roi_arrays = get_image_rois(roi_dict = roi_dict, img = img)
        #print(roi_arrays.keys())
        #print(roi_arrays["ROI_1"].shape)
        #Save
        np.savez_compressed(f"{clean_path}/{file_name[:-4]}", **roi_arrays, allow_pickle=True) # TODO need to re-run, but lets sort the filepaths and workflow first :)
        # break

if __name__ =="__main__":
    #this is abide data
    clean_scans(folder = 'ADHD200', target_folder='ADHD200_7', mask = 'Yeo2011_7Networks_MNI152_FreeSurferConformed1mm_LiberalMask.nii', mac=True)
    clean_scans(folder = 'ADHD200', target_folder='ADHD200_17', mask = 'Yeo2011_17Networks_MNI152_FreeSurferConformed1mm_LiberalMask.nii', mac=True)