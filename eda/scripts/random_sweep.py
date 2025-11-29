import warnings
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics import log_loss, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
from tqdm import  tqdm
import wandb


def load_x_y(dataset:str, binary = "Binary", demographic_features = "With_Demographic", node_features = "All_NF"):
    """ load_x_y returns X,y where X is a feature vector and y are the class labels, which features are included
    and if the class labels should be for the binary or multiclass task can be specified with the following:

    Args:
        dataset (str):  A dataset in {"train","test","val"}. 
        binary (str, optional): Return Binary or Multiclass Labels, choose a value in {"Binary", "Multiclass"} Defaults to "Binary".
        demographic_features (str, optional):  If demographic features should be included, choose between: {"With_Demographic","Without_Demographic"}. Defaults to "With_Demographic".
        node_features (str, optional): What node features should be included, {"No_NF","All_NF","No_PCA_NF","One_PCA_NF","All_PCA_NF"}. Defaults to "All_NF".

    Returns:
        X (ndarray): Feature vector
        y (ndarray): True Class Labels
    """
    MD_df = pd.read_csv("../notebooks/eda/Motion_and_Demographics.csv",index_col = 0)
    MD_df["Run"] = MD_df["scan_id"].str.split("_").str.get(-1)
    MD_df = MD_df.drop("scan_id",axis = 1)
    MD_df = MD_df.drop("Label",axis = 1)
    MD_df["Sex"] = MD_df["Sex"].map({"Male":0,"Female":1})
    MD_dict = MD_df.set_index(["Sub ID", "Dataset","Run"]).drop_duplicates().to_dict(orient="index")
    if binary == "Multiclass":
      class_dict = {'TD': 0, 'ASD-ADHD':1, 'ASD':2, 'ADHD':3}
    elif binary == "Binary":
      class_dict = {'TD': 0, 'ASD-ADHD':1, 'ASD':1, 'ADHD':1}

    file_list = pd.read_csv(f'../data.nosync/networks_multi/networks_multi/{dataset}_set_files.csv')['file'].to_list()
    data_list = []
    col_names = ["Sub ID","Run","dataset","y"]

    for i in tqdm(file_list):
        i = i.split('/')[2]
        network_class = class_dict[i.split('_')[3]]
        id =  i.split('_')[0]
        run = i.split('_')[1]
        dataset =  i.split('_')[2]
        G = nx.read_gml(f'../data.nosync/networks_multi/networks_multi/{i}')
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

          #features.append(node)
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
    return X,y

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
    if demographic_features == "Without_Demographic" and node_features == "No_NF":
        return [],[],[],[]
    X_train, y_train = load_x_y(dataset='train', binary= binary, demographic_features= demographic_features, node_features=node_features)   
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val, y_val = load_x_y(dataset='val', binary= binary, demographic_features= demographic_features, node_features=node_features)
    X_val = scaler.transform(X_val)
    X_test, y_test = load_x_y(dataset='test', binary= binary, demographic_features= demographic_features, node_features=node_features)
    X_test = scaler.transform(X_test)
    return X_train, y_train, X_val, y_val, X_test, y_test

def train_random(config=None, binary = False):
    """ function used by wandb to train and evaluate the random and prior baseline models 

    Args:
        config (_type_, optional): config for wandb. Defaults to None.
        binary (bool, optional): binary or multiclass task . Defaults to False.
    """

    # Train model, get predictions
    with wandb.init(config=config):
      # If called by wandb.agent, as below,
      # this config will be set by Sweep Controller
      config = wandb.config
      model = DummyClassifier(strategy= config.strategy,
                                random_state=rng)
      X_train, y_train, X_val, y_val, X_test, y_test = load_datasets(binary = config.binary, demographic_features = "With_Demographic", node_features = "No_PCA_NF")
      
      # just giving it any of the datasets (it only uses y anyway)
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

      f_1 = f1_score(y_pred ,y_val,average = "weighted")
      f_1_train = f1_score(y_pred_train ,y_train,average = "weighted")
      f_1_test = f1_score(y_pred_test ,y_test,average = "weighted")


      if binary == False:
        labels = ['TD', 'ASD-ADHD', 'ASD', 'ADHD']
      elif binary == True:
        labels = ["TD","ND"]

      print(n_ll,n_ll_train)
      wandb.log({
          "train_loss": n_ll_train,
          "val_loss": n_ll,
          "test_loss": n_ll_test,
          "val_f1_score": f_1,
          "train_f1_score": f_1_train,
          "test_f1_score": f_1_test
          })
      wandb.sklearn.plot_classifier(model,
                                  X_train, X_val,
                                  y_train, y_val,
                                  y_pred, y_probas,
                                  labels,
                                  is_binary=binary,
                                  model_name=f'Random_{config.strategy}')

def main_random():
    """ initialise a wandb run
    """
    wandb.init(project="random_model_with_scores_testing_script")
    is_binary = wandb.config.binary =="Binary"
    train_random(config=wandb.config, binary = is_binary)

if __name__ == "__main__":
    # 0: login to wandb
    wandb.login(key= "") # Replace with own key if running.

    # 1: Set Random Seed
    rng = np.random.RandomState(0)

    # 2: Define the search space
    sweep_configuration = {
        "method": "grid",
        "parameters": {
            "strategy": {"values": ['most_frequent', 'prior', 'stratified', 'uniform']},
            "binary": {"values":["Binary","Multiclass"]}
        }}

    # 3: Start the sweep
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="random_model_with_scores_testing_script")
    wandb.agent(sweep_id = sweep_id, function=main_random)
