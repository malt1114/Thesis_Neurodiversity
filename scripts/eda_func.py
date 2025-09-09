from matplotlib import pyplot as plt
from tqdm import tqdm
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
import math
import os
import re

def get_mean_and_var_stats(num_of_rois:str) -> pd.DataFrame:
    """This function calculates the mean and variance
       of the ROIs pr. subject and saves them as a CSV file.

    Args:
        data_folder (str): The folder where the data is located (data/clean/XXXXXXX)
        mac (bool, optional): If the path is on a mac. Defaults to True.

    Returns:
        pd.DataFrame: the dataframes one for means and one for variance
    """

    meta_data = pd.read_csv(f'data.nosync/phenotypic/subjects_with_meta_{num_of_rois}.csv',
                            index_col= 'Unnamed: 0')
    meta_data['Co-Diagnosis'] = meta_data['Co-Diagnosis'].apply(lambda x: str(x))
    meta_data['Full Diagnosis'] = meta_data['Diagnosis'] + '+' + meta_data['Co-Diagnosis']
    meta_data['Full Diagnosis'] = meta_data['Full Diagnosis'].apply(lambda x: x.replace('+nan','').replace('TD+Other', 'TD'))

    mean_pr_roi = []
    var_pr_roi = []

    for dataset in meta_data.Dataset.unique():
        temp_data = meta_data[meta_data['Dataset'] == dataset]

        for idx, row in temp_data.iterrows(): #For each subject
            subject_means = {}
            subject_variance = {}
            
            #Create meta keys
            subject_means['subject'], subject_means['dataset'], subject_means['diagnosis'] = row["Sub ID"], row["Dataset"], row['Full Diagnosis']
            subject_variance['subject'], subject_variance['dataset'], subject_variance['diagnosis'] = row["Sub ID"], row["Dataset"], row['Full Diagnosis']
        
            subject = np.load(row['file_path'], allow_pickle = True)
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
    mean_pr_roi.to_csv(f'data.nosync/stats/{num_of_rois}_mean_subject.csv', sep= ";")
    
    #make into dataframe, exstract subject info and save stats
    var_pr_roi = pd.DataFrame(var_pr_roi)
    var_pr_roi.to_csv(f'data.nosync/stats/{num_of_rois}_variance_subject.csv', sep= ";")

    return mean_pr_roi, var_pr_roi

def test_dist(data: pd.DataFrame, regions:list[str], thres = 0.05) -> None:
    """This function test if the values of the ROI are from 
       the same distribution. It does it for both gender and 
       the diagnosis

    Args:
        data (pd.DataFrame): the data which to use
        regions (list[str]): a list of regions to test
    """
    #Gender
    p_values = []
    results = []
    for roi in regions:
        a, b = data[data['Sex'] == 'Male'][roi], data[data['Sex'] == 'Female'][roi]
        p_value = float(stats.ttest_ind(a, b, 
                                        equal_var = False, 
                                        alternative='two-sided').pvalue)
        p_values.append(p_value)
        results.append([roi, p_value])
        
    #Make FDR
    fdr = stats.false_discovery_control(p_values)
    for i in range(len(results)):
        results[i].append(fdr[i])

    #print results
    for i in results:
        print(f'{i[0]} (Between sex): The P-value is {i[1]}, after FDR {i[2]}')

    print('#'*50)
    p_values = []
    results = []
    #Diagnosis
    for roi in regions:
        dia = data['diagnosis'].unique().tolist()
        sampels = [data[data['diagnosis'] == i][roi] for i in dia]
        p_value = float(stats.f_oneway(*sampels).pvalue)
        p_values.append(p_value)
        results.append([roi, p_value])
    
    #Make FDR
    fdr = stats.false_discovery_control(p_values)
    for i in range(len(results)):
        results[i].append(fdr[i])

    #print results
    for i in results:
        print(f'{i[0]} (Between diagnosis): The P-value is {i[1]}, after FDR {i[2]}')

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
    x_min = min(data[regions].min().to_list())
    x_max = max(data[regions].max().to_list())

    #Sub plot size
    sub_size = math.ceil(math.sqrt(len(regions)))
    row = round(math.ceil(len(regions)/sub_size))

    #Make subplot
    fig, axs = plt.subplots(sub_size, row, figsize=(15, 15), tight_layout=True)
    fig.suptitle(title, fontsize = 20)

    temp_rois = regions.copy()

    #Store y axis
    y_max = []
    y_min = []

    for x in range(sub_size):
        for y in range(row):
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