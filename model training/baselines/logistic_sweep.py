import numpy as np
import pandas as pd
import networkx as nx
from tqdm import  tqdm
import wandb
import matplotlib.pyplot as plt 
import joblib
import warnings
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay,roc_curve, auc, log_loss, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
import matplotlib
matplotlib.use('agg')

def load_x_y(dataset:str,binary = "Binary", demographic_features = "With_Demographic", node_features = "All_NF"):
    """ load_x_y returns X,y,col_names where X is a feature vector, y are the class labels, and col_names are the column names of the feature vector. 
    Which features are included and if the class labels should be for the binary or multiclass task can be specified with the following:

    Args:
        dataset (str):  A dataset in {"train","test","val"}. 
        binary (str, optional): Return Binary or Multiclass Labels, choose a value in {"Binary", "Multiclass"} Defaults to "Binary".
        demographic_features (str, optional):  If demographic features should be included, choose between: {"With_Demographic","Without_Demographic"}. Defaults to "With_Demographic".
        node_features (str, optional): What node features should be included, {"No_NF","All_NF","No_PCA_NF","One_PCA_NF","All_PCA_NF"}. Defaults to "All_NF".

    Returns:
        X (ndarray): Feature vector
        y (ndarray): True Class Labels
        col_names(list): Column Names of the Feature vector
    """
    MD_df = pd.read_csv("../../eda/notebooks/Motion_and_Demographics.csv",index_col = 0)
    MD_df["Run"] = MD_df["scan_id"].str.split("_").str.get(-1)
    MD_df = MD_df.drop("scan_id",axis = 1)
    MD_df = MD_df.drop("Label",axis = 1)
    MD_df["Sex"] = MD_df["Sex"].map({"Male":0,"Female":1})
    MD_dict = MD_df.set_index(["Sub ID", "Dataset","Run"]).drop_duplicates().to_dict(orient="index")    
    if binary == "Multiclass":
        class_dict = {'TD': 0, 'ASD-ADHD':1, 'ASD':2, 'ADHD':3}
    elif binary == "Binary":
        class_dict = {'TD': 0, 'ASD-ADHD':1, 'ASD':1, 'ADHD':1}
    
    file_list = pd.read_csv(f'../../data.nosync/networks_multi/networks_multi/{dataset}_set_files.csv')['file'].to_list()

    data_list = []
    col_names = ["Sub ID","Run","dataset","y"]

    for i in tqdm(file_list):
        i = i.split('/')[2]
        network_class = class_dict[i.split('_')[3]]
        id =  i.split('_')[0]
        run = i.split('_')[1]
        dataset =  i.split('_')[2]
        for part in [1]:
          G = nx.read_gml(f'../../data.nosync/networks_multi_correct/networks_multi/{i}')
          features = [id,run,dataset,network_class]

          n_f_names = G.nodes["1"].keys()


          if node_features == "All_NF":
            n_f_names = n_f_names
          elif node_features == "No_PCA_NF":
            n_f_names = [name for name in n_f_names if  "pca" not in name]
          elif node_features == "All_PCA_NF":
            n_f_names = [name for name in n_f_names if  "pca" in name]
          elif node_features == "One_PCA_NF":
            n_f_names = [name for name in n_f_names if  "pca_all" in name]
          elif node_features == "No_NF":
            n_f_names = []
          for n in G.nodes():
            for name in n_f_names:
              features.append(G.nodes[n][name])

          if demographic_features == "With_Demographic":
            for demographic_info in  MD_dict[(int(id),dataset,run)].values():
              features.append(demographic_info)

          data_list.append(features)
    if node_features != "No_NF":
      for n in G.nodes():
        col_names.extend([i+"_"+n for i in n_f_names])
    elif node_features == "No_NF":
      pass


    if demographic_features == "With_Demographic":
      col_names.extend(MD_dict[(int(id),dataset,run)].keys())

    data_list = pd.DataFrame(data_list,columns=col_names)
    X =  np.array(data_list[col_names[4:]])
    y =  np.array(data_list["y"])
    return X,y,col_names

def load_datasets(binary = "Binary", demographic_features = "With_Demographic", node_features = "All_NF"):
    """Calls load_x_y to generate X,y for train, val and test, and scales them with a standard scalar.

    Args:
        binary (str, optional): Return Binary or Multiclass Labels, choose a value in {"Binary", "Multiclass"} Defaults to "Binary".
        demographic_features (str, optional):  If demographic features should be included, choose between: {"With_Demographic","Without_Demographic"}. Defaults to "With_Demographic".
        node_features (str, optional): What node features should be included, {"No_NF","All_NF","No_PCA_NF","One_PCA_NF","All_PCA_NF"}. Defaults to "All_NF".

    Returns:
        X_train (ndarray): Feature vector for train
        y_train (ndarray): True Class Labels for train
        X_val (ndarray): Feature vector for val
        y_val (ndarray): True Class Labels for val
        X_test (ndarray): Feature vector for test
        y_test (ndarray): True Class Labels for test
            
    """
    print("Loading: ", binary,demographic_features,node_features)
    if demographic_features == "Without_Demographic" and node_features == "No_NF":
        return [],[],[],[]
    X_train, y_train,col_names = load_x_y(dataset='train', binary= binary, demographic_features= demographic_features, node_features=node_features)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val, y_val,col_names = load_x_y(dataset='val', binary= binary, demographic_features= demographic_features, node_features=node_features)
    X_val = scaler.transform(X_val)
    X_test, y_test, col_names = load_x_y(dataset='test', binary= binary, demographic_features= demographic_features, node_features=node_features)
    X_test = scaler.transform(X_test)
    return X_train, y_train, X_val, y_val, X_test, y_test,col_names

def plot_cm_image(y_true, y_pred, labels, title):
    """Generate a W&B-safe confusion matrix image."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap='Blues', ax=ax, colorbar=False)
    ax.set_title(title)
    plt.tight_layout()
    img = wandb.Image(fig)
    plt.close(fig)
    return img

def plot_roc_image(y_true, y_probas, labels, title):
    """Generate a W&B-safe ROC curve image for multiclass or binary classification."""
    fig, ax = plt.subplots()
    
    # Handle binary vs multiclass cases
    if y_probas.ndim == 1 or y_probas.shape[1] == 1:
        fpr, tpr, _ = roc_curve(y_true, y_probas)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    else:
        for i, label in enumerate(labels):
            fpr, tpr, _ = roc_curve(y_true == label, y_probas[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=2, label=f"{label} (AUC = {roc_auc:.2f})")

    ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc="lower right")
    plt.tight_layout()
    img = wandb.Image(fig)
    plt.close(fig)
    return img

def train_logistic(config=None, binary = False):
    """ function used by wandb to train and evaluate the logistic models 

    Args:
        config (_type_, optional): config for wandb. Defaults to None.
        binary (bool, optional): binary or multiclass task . Defaults to False.
    """
    # Train model, get predictions
    with wandb.init(config=config):
      # If called by wandb.agent, as below,
      # this config will be set by Sweep Controller
      config = wandb.config
      model = LogisticRegression(penalty = config.penalty,
                                tol = config.tol,
                                C = config.C,
                                class_weight = config.class_weight,
                                solver = config.solver,
                                max_iter = config.max_iter, random_state=rng)
      X_train, y_train, X_val, y_val, X_test, y_test,col_names = dataset_dict[config.node_features][config.binary][config.demographic][0]
      model.fit(X_train, y_train)

      y_pred = model.predict(X_val)
      y_pred_test = model.predict(X_test)
      y_pred_train = model.predict(X_train)

      y_probas = model.predict_proba(X_val)
      y_probas_train = model.predict_proba(X_train)
      y_probas_test = model.predict_proba(X_test)


        
      n_ll = log_loss(y_val ,y_probas)
      n_ll_train = log_loss(y_train ,y_probas_train)
      n_ll_test = log_loss(y_test ,y_probas_test)

      f_1_val_macro = f1_score(y_pred= y_pred ,y_true = y_val,average = "macro")
      f_1_train_macro = f1_score(y_pred = y_pred_train ,y_true = y_train,average = "macro")
      f_1_test_macro = f1_score(y_pred = y_pred_test ,y_true = y_test,average = "macro")

      f_1_val_mic = f1_score(y_pred = y_pred ,y_true =y_val,average = "micro")
      f_1_train_mic = f1_score(y_pred = y_pred_train ,y_true =y_train,average = "micro")
      f_1_test_mic = f1_score(y_pred =y_pred_test ,y_true =y_test,average = "micro")


      if binary == False:
        labels = ['TD', 'ASD-ADHD', 'ASD', 'ADHD']
      elif binary == True:
        labels = ["TD","ND"]


      importances = model.coef_
      #indices = np.argsort(importances)[::-1]
      print(n_ll,n_ll_train)
      wandb.log({
          "train_loss": n_ll_train,
          "val_loss": n_ll,
          "test_loss": n_ll_test,
          "val_f1_score_macro": f_1_val_macro,
          "train_f1_score_macro": f_1_train_macro,
          "test_f1_score_macro": f_1_test_macro,
          "val_accuracy": f_1_val_mic,
          "train_accuracy": f_1_train_mic,
          "test_accuracy": f_1_test_mic,
          # --- Test Plots ---
          "Test Confusion Matrix": plot_cm_image(y_test, y_pred_test, labels, "Test Confusion Matrix"),
          # "Test ROC": plot_roc_image(y_test, y_probas_test, labels, "Test ROC Curve")
      })

      wandb.log({
          # --- Validation Plots ---
          "Val Confusion Matrix": plot_cm_image(y_val, y_pred, labels, "Validation Confusion Matrix"),
          # "Val ROC": plot_roc_image(y_val, y_probas, labels, "Validation ROC Curve")
      })

      wandb.log({
          # --- Training Plots ---
          "Train Confusion Matrix": plot_cm_image(y_train, y_pred_train, labels, "Train Confusion Matrix"),
          # "Train ROC": plot_roc_image(y_train, y_probas_train, labels, "Train ROC Curve")
      })


      wandb.sklearn.plot_classifier(model,
                                  X_train, X_val,
                                  y_train, y_val,
                                  y_pred, y_probas,
                                  labels,
                                  is_binary=binary,
                                  model_name='LogisticRegression')
      #joblib.dump(model, f"logistic_models/model_{config.node_features}_{config.binary}_{config.demographic}_{config.penalty}_{config.C}_{config.solver}.pkl")

def main_logistic():
    wandb.init(project="logistic_sweep_split_simple")
    is_binary = wandb.config.binary =="Binary"
    train_logistic(config=wandb.config, binary = is_binary)

if __name__ == "__main__":
    # 0: preload data for faster data-loading in the sweeps

    # # Define possibilities for datasets
    # Binary = ["Binary","Multiclass"]
    # Demographic = ["With_Demographic","Without_Demographic"]
    # #Node_Features = ["All_NF","No_PCA_NF","All_PCA_NF","One_PCA_NF","No_NF"] # replace line below with this to run sweep with pca features
    # Node_Features = ["No_NF","No_PCA_NF"]

    # # create dictionary for fast access to datasets
    # dataset_dict = {i:{
    #     j:{
    #       k:[load_datasets(binary = j, demographic_features = k, node_features = i)] for k in Demographic}
    #     for j in Binary }
    #                 for i in Node_Features}
    # np.save('../../eda/notebooks/dataset_dict.npy', dataset_dict) 
    
    # 0: or preload pre-loaded dictionary from save
    dataset_dict = np.load('../../eda/notebooks/dataset_dict.npy', allow_pickle=True).item()
  
    # uncomment and replace the below line with the one above to run with the split scan data instead
    # dataset_dict = np.load('dataset_dict_split.npy', allow_pickle=True).item()

    # # 0: login to wandb
    wandb.login(key= "") # Replace with own key if running.
    # 1: Set Random Seed
    rng = np.random.RandomState(0)


    # 2: Define the search space
    sweep_configuration = {
        "method": "grid",
        "metric": {"goal": "minimize", "name": "val_loss"},
        "parameters": {
            "penalty": {"values": ["l2", "l1", "elasticnet"]},
            "tol": {"values": [0.0001]},#1e-3, 1e-4, 1e-5]},  use default
            "C":{"values": [1, 0.1, 0.01,1e-3]},
            "class_weight":{"values":["balanced"]},
            "solver":{"values":["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"]},
            "max_iter":{"values":[50]}, # 50 was found to be best[10,50,100,250,500]
            ## which data demographic and node features
            "demographic": {"values":["With_Demographic","Without_Demographic"]},
            "node_features": {"values":["No_NF","No_PCA_NF"]},
            "binary": {"values":["Binary","Multiclass"]}
        }
    }
    # 3: Start the sweep
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="logistic_test_confusion_roc_testing_script")
    wandb.agent(sweep_id = sweep_id, function=main_logistic)
