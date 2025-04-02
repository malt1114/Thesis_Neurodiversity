from matplotlib import pyplot as plt
from tqdm import tqdm
import seaborn as sns
import pandas as pd
import numpy as np
import scipy
import math
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

def test_dist(data: pd.DataFrame, regions:list[str]) -> None:
    """This function test if the values of the ROI are from 
       the same distribution. It does it for both gender and 
       the diagnosis

    Args:
        data (pd.DataFrame): the data which to use
        regions (list[str]): a list of regions to test
    """
    #Gender
    for roi in regions:
        a, b = data[data['Gender'] == 'Male'][roi], data[data['Gender'] == 'Female'][roi]
        p_value = float(scipy.stats.ttest_ind(a, b, 
                                            equal_var = False, 
                                            alternative='two-sided').pvalue)
        print(f'{roi} (Gender): The P-value is {p_value}')

    print('#'*50)
    #Diagnosis
    for roi in regions:
        dia = data['DX'].unique().tolist()
        sampels = [data[data['DX'] == i][roi] for i in dia]
        p_value = float(scipy.stats.f_oneway(*sampels).pvalue)
        print(f'{roi} (Diagnosis): The P-value is {p_value}')


def plot_small_multiple_rois(data:pd.DataFrame, regions:list[str], title:str, hue_col:str) -> None:
    """This fucntion plot small multiple dist plots
       for each of the regions. Furthermore the hue, 
       can be 

    Args:
        data (pd.DataFrame): _description_
        regions (list[str]): _description_
        title (str): _description_
        hue_col (str): _description_
    """
    #Get axis limits to uniform x-axis
    x_min = min(data[regions].min().to_list())-0.1
    x_max = max(data[regions].max().to_list())+0.1

    #Sub plot size
    sub_size = math.ceil(math.sqrt(len(regions)))

    #Make subplot
    fig, axs = plt.subplots(sub_size, sub_size, figsize=(10, 10), tight_layout=True)
    fig.suptitle(title, fontsize = 20)

    temp_rois = regions.copy()

    #Store y axis
    y_max = []
    y_min = []

    for x in range(sub_size):
        for y in range(sub_size):
            #If no ROIs left break
            if len(temp_rois) == 0:
                break
            #If top-right add legend and move to the right
            add_legend = True if [x,y] == [0,sub_size-1] else False
            
            #Create subplot
            sns.kdeplot(data = data, 
                        x = temp_rois[0], 
                        hue= hue_col, 
                        ax = axs[x, y], 
                        legend= add_legend)
            
            #Set title and x-axis range 
            axs[x, y].set_title(temp_rois[0])

            if add_legend:
                sns.move_legend(axs[x, y], 
                                "upper left", 
                                bbox_to_anchor=(1, 1))

            #Keep track of the y axis -> so it can be uniformed
            lim = axs[x,y].get_ylim()
            y_max.append(lim[1])
            y_min.append(lim[0])
            temp_rois.remove(temp_rois[0])

    for ax in axs.flat:
        #Set axis limits and labels
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(min(y_min), max(y_max))
        ax.set(xlabel='Activity', ylabel='Density')

    # Hide x labels and tick labels for top plots and y ticks for right plots
    for ax in axs.flat:
        ax.label_outer()

def group_roi_heat_map(data:pd.DataFrame, group:str, title:str, regions:list[str]) -> None:
    """_summary_

    Args:
        data (pd.DataFrame): _description_
        group (str): _description_
        title (str): _description_
        regions (list[str]): _description_
    """
    #Remove nans from category
    data = data[~data[group].isna()]
    #Get unique groups
    groups = data[group].unique()
    #Make subplot and add title
    fig, axs = plt.subplots(len(groups), figsize=(10, 10), tight_layout=True)
    fig.suptitle(title, fontsize = 20)
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, 
                        top=1, wspace=0.4, hspace=0.4)
    #Get min and max for colors
    x_min = min(data[regions].min())
    x_max = max(data[regions].max())
    #Create color map
    color_map = sns.color_palette("Blues", as_cmap=True)
    #Plot the heatmaps
    for x in range(len(groups)):
        temp_data = data[data[group] == groups[x]]
        #Create subplot
        sns.heatmap(temp_data.loc[:, regions],
                    vmin = x_min,
                    vmax = x_max,
                    cmap = color_map, 
                    ax = axs[x])
        axs[x].set_title(groups[x])