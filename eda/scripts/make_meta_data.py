import pandas as pd
import numpy as np
import os

def create_meta_data_file(path_addon:str):
    #Read data
    abidei = pd.read_csv(f'../../data{path_addon}/phenotypic/ABIDEI/ABIDEI_phenotypic_NYU.csv')
    abideii = pd.read_csv(f'../../data{path_addon}/phenotypic/ABIDEII/ABIDEII-NYU_1.csv')
    adhd_200 = pd.read_csv(f'../../data{path_addon}/phenotypic/ADHD200/NYU2025_full_corrected.csv')
    adhd_200_missing = pd.read_csv(f'../../data{path_addon}/phenotypic/ADHD200/MissingParticipants.csv')
    adhd_200_missing = adhd_200_missing.rename({'ID':'ScanDir ID'}, axis = 1)
    adhd_200 = pd.concat([adhd_200, adhd_200_missing])
    print(adhd_200.columns)

    #CLEAN ABIDEI META
    #Select columns
    abidei = abidei[['SUB_ID', 'AGE_AT_SCAN', 'SEX', 'FIQ', 'DX_GROUP', 'COMORBIDITY']]
    #Map values
    abidei['SEX'] = abidei['SEX'].replace({1:'Male', 2: 'Female'})
    abidei['DX_GROUP'] = abidei['DX_GROUP'].replace({1: 'ASD', 2:'TD'})
    abidei['COMORBIDITY'] = abidei['COMORBIDITY'].apply(lambda x: 'ADHD' if 'ADHD' in str(x).upper()
                                                        else x)
    abidei['COMORBIDITY'] = abidei['COMORBIDITY'].apply(lambda x: 'Other' if str(x) != 'ADHD' and str(x) != 'nan' 
                                                        else x)
    abidei['COMORBIDITY'] = abidei['COMORBIDITY'].apply(lambda x: None if str(x) == 'nan' 
                                                        else x)
    abidei['SUB_ID'] = abidei['SUB_ID'].apply(lambda x: str(x).zfill(7))

    abidei = abidei.rename({'SUB_ID': 'Sub ID',
                            'AGE_AT_SCAN': 'Age',
                            'SEX': 'Sex',
                            'FIQ': 'IQ',
                            'DX_GROUP': 'Diagnosis',
                            'COMORBIDITY': 'Co-Diagnosis'}, axis = 1)
    abidei['Dataset'] = 'ABIDEI'

    #CLEAN ABIDEII META
    #Select columns
    abideii = abideii[['SUB_ID', 'AGE_AT_SCAN', 'SEX', 'FIQ', 'DX_GROUP', 'NONASD_PSYDX_LABEL']]
    #Map values
    abideii['SEX'] = abideii['SEX'].replace({1:'Male', 2: 'Female'})
    abideii['DX_GROUP'] = abideii['DX_GROUP'].replace({1: 'ASD', 2:'TD'})
    abideii['NONASD_PSYDX_LABEL'] = abideii['NONASD_PSYDX_LABEL'].apply(lambda x: 'ADHD' if 'ADHD' in str(x).upper()
                                                        else x)
    abideii['NONASD_PSYDX_LABEL'] = abideii['NONASD_PSYDX_LABEL'].apply(lambda x: 'Other' if str(x) != 'ADHD' and str(x) != 'none' 
                                                        else x)
    abideii['NONASD_PSYDX_LABEL'] = abideii['NONASD_PSYDX_LABEL'].apply(lambda x: '' if str(x) == 'nan' 
                                                        else x)
    abideii['SUB_ID'] = abideii['SUB_ID'].apply(lambda x: str(x).zfill(7))

    abideii = abideii.rename({'SUB_ID': 'Sub ID',
                            'AGE_AT_SCAN': 'Age',
                            'SEX': 'Sex',
                            'FIQ': 'IQ',
                            'DX_GROUP': 'Diagnosis',
                            'NONASD_PSYDX_LABEL': 'Co-Diagnosis'}, axis = 1)
    abideii['Dataset'] = 'ABIDEII'

    #CLEAN ADHD 200 META
    adhd_200['Gender'] = adhd_200['Gender'].replace({0: 'Female', 1:'Male'})
    adhd_200['Secondary Dx '] = adhd_200['Secondary Dx '].apply(lambda x: 'ASD' if 'ASD' in str(x) else x)
    adhd_200['Secondary Dx '] = adhd_200['Secondary Dx '].apply(lambda x: 'Other' if 'ASD' not in str(x) and str(x) != 'nan' else x)
    adhd_200['Secondary Dx '] = adhd_200['Secondary Dx '].apply(lambda x: '' if str(x) == 'nan' 
                                                                        else x)
    adhd_200['DX'] = adhd_200['DX'].apply(lambda x: 'TD' if x == 0 else 'ADHD')
    adhd_200['ScanDir ID'] = adhd_200['ScanDir ID'].apply(lambda x: str(x).zfill(7))

    adhd_200 = adhd_200.rename({'ScanDir ID': 'Sub ID',
                            'Gender': 'Sex',
                            'Full4 IQ': 'IQ',
                            'DX': 'Diagnosis',
                            'Secondary Dx ': 'Co-Diagnosis'}, axis = 1)
    adhd_200 = adhd_200.replace({-999.0: np.nan})
    adhd_200 = adhd_200[['Sub ID', 'Age', 'Sex', 'IQ', 'Diagnosis', 'Co-Diagnosis']]
    adhd_200['Dataset'] = 'ADHD200'

    #CONCAT AND SAVE
    data = pd.concat([abidei, abideii, adhd_200]).reset_index(drop= True)
    data['Co-Diagnosis'] = data['Co-Diagnosis'].replace({'none': ''})
    data.to_csv(f'../data{path_addon}/phenotypic/meta_data.csv')


def link_meta_to_files(num_of_roi:str, path_addon: str):
    abidei = f'../../data{path_addon}/clean/ABIDEI_{num_of_roi}'
    abidei = os.listdir(abidei)
    abidei = [[i.split('_')[0], 'ABIDEI', f'data.nosync/clean/ABIDEI_{num_of_roi}/'+i] for i in abidei]

    abideii = f'../../data{path_addon}/clean/ABIDEII_{num_of_roi}'
    abideii = os.listdir(abideii)
    abideii = [[i.split('_')[0], 'ABIDEII', f'data.nosync/clean/ABIDEII_{num_of_roi}/'+i] for i in abideii]

    adhd200 = f'../../data{path_addon}/clean/ADHD200_{num_of_roi}'
    adhd200 = os.listdir(adhd200)
    adhd200 = [[i.split('_')[0], 'ADHD200', f'data.nosync/clean/ADHD200_{num_of_roi}/'+i] for i in adhd200]

    #Create dataframe
    all_files = abidei+abideii+adhd200
    data = pd.DataFrame(all_files, columns=['Sub ID', 'Dataset', 'file_path'])
    data['Sub ID'] = data['Sub ID'].apply(lambda x: x[4:].zfill(7))

    #Add meta data
    meta_data = pd.read_csv('../../data.nosync/phenotypic/meta_data.csv', index_col = 'Unnamed: 0')
    meta_data['Sub ID'] = meta_data['Sub ID'].apply(lambda x: str(x).zfill(7))

    result = pd.merge(data, meta_data, how="left", on=['Sub ID', 'Dataset'])
    result.to_csv(f'../../data{path_addon}/phenotypic/subjects_with_meta_{num_of_roi}.csv')

if __name__ =="__main__":
    create_meta_data_file(path_addon = ".nosync")
    link_meta_to_files('7', path_addon = ".nosync")
    link_meta_to_files('17',path_addon = ".nosync")