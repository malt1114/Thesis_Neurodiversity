from nilearn import image as nimg
from tqdm import tqdm
import numpy as np
import os


def get_roi_dict(mask: np.array) -> dict[int:np.array]:
    """This function returns a dict of the ROI masks

    Returns:
        dict: key is the roi number and value is a np array of bools
    """
    #Get number of unique ROIs
    num_of_roi = np.unique(mask).tolist()
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

def clean_scans(folder: str, mask) -> None:
    """This function takes a folder (str) that is places
       in data.nosync/clean, and cleans it into 
       numpy files.

    Args:
        folder (str): the folder with files to clean
    """
    preprocessed_path = f'../data.nosync/preprocessed/{folder}'
    clean_path = f'../data.nosync/clean/{folder}'

    mask = nimg.load_img(f"../data.nosync/{mask}")

    file_list = os.listdir(preprocessed_path)
    for file_name in tqdm(file_list):
        #Load image
        img = nimg.load_img(f"{preprocessed_path}/{file_name}")
        
        #Fit mask to image
        fitted_mask = nimg.resample_img(mask, 
                                        target_affine=img.affine, 
                                        interpolation='nearest',
                                        target_shape=img.shape[:3], 
                                        force_resample = True,
                                        copy_header=True)
        #Make into numpy arrays
        fitted_mask = fitted_mask.get_fdata()
        img = img.get_fdata()

        #Get ROI
        roi_dict = get_roi_dict(mask = fitted_mask)

        #Get roi arrays over time
        roi_arrays = get_image_rois(roi_dict = roi_dict, img = img)

        #Save
        np.savez_compressed(f"{clean_path}/{file_name[:-4]}", **roi_arrays, allow_pickle=True)


clean_scans(folder = 'NYU', mask = 'Yeo2011_7Networks_MNI152_FreeSurferConformed1mm_LiberalMask.nii')