# Graph-Based fMRI Representations: Exploring Differentiation of Neurodiversity
This is the code base for the thesis "Graph-Based fMRI Representations: Exploring Differentiation of Neurodiversity". Created by Malthe Pabst and Nicola Clark.

# Data Sources

ABIDE I: s3://fcp-indi/data/Projects/ABIDE/RawDataBIDS/NYU

ABIDE II: s3://fcp-indi/data/Projects/ABIDE2/RawData/ABIDEII-NYU_1

ADHD-200: s3://fcp-indi/data/Projects/ADHD200/RawDataBIDS/NYU

# C-PAC
The data used in this paper, originates from the data sources above. However, due to storage limits on GitHub, the data would have to be processed locally. As such, use the default pipeline in the "C-PAC" folder, along side the C-PAC Docker or Singularity container.

For more information about C-PAC and how to set up see: https://fcp-indi.github.io/docs/latest/user/quick.html

# File Structure
```
.
├── C-PAC
│   ├── Create job files.ipynb / Creates the SLURM jobs
│   ├── default_pipeline_changed.yml / the C-PAC preprocessing pipeline
│   └── jobs/ the SLURM jobs for the C-PAC preprocessing
│
├── data.nosync
│   ├── networks_multi_gat.zip / GAT networks
│   ├── networks_multi.zip/ The multi edge networks
│   ├── phenotypic/ phenotypic data for the participants
│   ├── stats.zip/ statistics of the fMRI scans
│   ├── two_scans_true.csv/ list of participants with two scans
│   ├── Yeo2011_7Networks_MNI152_FreeSurferConformed1mm_LiberalMask.nii.gz / network parcellation mask 7-network
│   └── Yeo2011_17Networks_MNI152_FreeSurferConformed1mm_LiberalMask.nii.gz/ network parcellation mask 17-network
│
├── Diagrams/ The diagrams used in the paper
│
├── eda
│   ├── latex tabels/ notebook for creating dataset statistics for the paper
│   ├── notebooks/ eda notebooks
│   ├── pics/ pictures used in the paper
│   └── scripts/ scripts that is used in the eda, for creating the networks, preprocessing the fmri data after the C-PAC pipeline and scripts for creating statistics.
│
├── legacy code/ old notebooks and scripts
│
├── model training
│   ├── 1. Create train_test_val.ipynb/ creating the train-valdiation-test split
│   ├── 2. Random models.ipynb/ random and prior models
│   ├── 3. Evaluation.ipynb/ evaluation of the models
│   ├── baselines/ the baseline logistic and random models scripts
│   ├── GAT Scripts/ the scripts used for the GAT sweep
│   ├── GCN scripts/ the scripts used for the GCN sweep
│   ├── help_funcs/ different scripts with help functions for model traning and evaluation
│   ├── models/ GAT and GCN architecture scripts
│   ├── pics/ evaluation pictures
│   └── saved_models/ the best GAT and GCN models + hyperparameters
│
└── requirements.txt/ requirements for python enviorment
```

## Use of Generative AI: 
ChatGPT was used to aid in some of the plotting with Seaborne and Matplotlib, raw outputs were ammended and verified by the authors of this GitHub to ensure correctness. Example input: "Please change the font size on this graph".
