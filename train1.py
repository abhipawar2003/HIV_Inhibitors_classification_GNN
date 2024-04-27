import torch 
from torch_geometric.data import DataLoader
from sklearn.metrics import confusion_matrix, f1_score, \
    accuracy_score, precision_score, recall_score, roc_auc_score
import numpy as np
from tqdm import tqdm
from dataset_featurizer import MoleculeDataset
from model import GNN
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd 
import random  
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# def train_one_epoch(epoch, model, train_loader, optimizer, loss_fn):
#     all_preds = []
#     all_labels = []
#     running_loss = 0.0
#     step = 0
#     for _, batch in enumerate(tqdm(train_loader)):
#         batch.to(device)  
#         optimizer.zero_grad() 
#         pred = model(batch.x.float(), 
#                                 batch.edge_attr.float(),
#                                 batch.edge_index, 
#                                 batch.batch) 
#         loss = loss_fn(torch.squeeze(pred), batch.y.float())
#         loss.backward()  
#         optimizer.step()  
#         running_loss += loss.item()
#         step += 1
#         all_preds.append(np.rint(torch.sigmoid(pred).cpu().detach().numpy()))
#         all_labels.append(batch.y.cpu().detach().numpy())
#     all_preds = np.concatenate(all_preds).ravel()
#     all_labels = np.concatenate(all_labels).ravel()
#     calculate_metrics(all_preds, all_labels, epoch, "train")
#     return running_loss/step

def train_one_epoch(epoch, model, train_loader, optimizer, loss_fn):
    # Enumerate over the data
    all_preds = []
    all_labels = []
    running_loss = 0.0
    step = 0
    for _, batch in enumerate(tqdm(train_loader)):
        # Use GPU
        batch.to(device)  

        #######################
        # Add debugging information here
        print("Batch shape:", batch.x.shape)
        print("Batch type:", type(batch.x))
        #######################

        # Reset gradients
        optimizer.zero_grad() 
        # Passing the node features and the connection info
        pred = model(batch.x.float(), 
                                batch.edge_attr.float(),
                                batch.edge_index, 
                                batch.batch) 
        # Calculating the loss and gradients
        loss = loss_fn(torch.squeeze(pred), batch.y.float())
        loss.backward()  
        optimizer.step()  
        # Update tracking
        running_loss += loss.item()
        step += 1
        all_preds.append(np.rint(torch.sigmoid(pred).cpu().detach().numpy()))
        all_labels.append(batch.y.cpu().detach().numpy())
    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()
    calculate_metrics(all_preds, all_labels, epoch, "train")
    return running_loss/step


def test(epoch, model, test_loader, loss_fn):
    all_preds = []
    all_preds_raw = []
    all_labels = []
    running_loss = 0.0
    step = 0
    for batch in test_loader:
        batch.to(device)  
        pred = model(batch.x.float(), 
                        batch.edge_attr.float(),
                        batch.edge_index, 
                        batch.batch) 
        loss = loss_fn(torch.squeeze(pred), batch.y.float())
        running_loss += loss.item()
        step += 1
        all_preds.append(np.rint(torch.sigmoid(pred).cpu().detach().numpy()))
        all_preds_raw.append(torch.sigmoid(pred).cpu().detach().numpy())
        all_labels.append(batch.y.cpu().detach().numpy())
    
    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()
    calculate_metrics(all_preds, all_labels, epoch, "test")
    log_conf_matrix(all_preds, all_labels, epoch)
    return running_loss/step

def log_conf_matrix(y_pred, y_true, epoch):
    cm = confusion_matrix(y_pred, y_true)
    classes = ["0", "1"]
    df_cfm = pd.DataFrame(cm, index = classes, columns = classes)
    plt.figure(figsize = (10,7))
    cfm_plot = sns.heatmap(df_cfm, annot=True, cmap='Blues', fmt='g')
    plt.show()

def calculate_metrics(y_pred, y_true, epoch, type):
    print(f"\n Confusion matrix: \n {confusion_matrix(y_pred, y_true)}")
    print(f"F1 Score: {f1_score(y_true, y_pred)}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred)}")
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    print(f"Precision: {prec}")
    print(f"Recall: {rec}")
    try:
        roc = roc_auc_score(y_true, y_pred)
        print(f"ROC AUC: {roc}")
    except:
        print(f"ROC AUC: not defined")

# %% Run the training

def run_one_training(params):
    # Loading the dataset
    print("Loading dataset...")
    train_dataset = MoleculeDataset(root="data/", filename="HIV_train_oversampled.csv")
    test_dataset = MoleculeDataset(root="data/", filename="HIV_test.csv", test=True)
    params["model_edge_dim"] = train_dataset[0].edge_attr.shape[1]

    # Prepare training
    train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=params["batch_size"], shuffle=True)

    # Loading the model
    print("Loading model...")
    model_params = {k: v for k, v in params.items() if k.startswith("model_")}
    model = GNN(feature_size=train_dataset[0].x.shape[1], model_params=model_params) 
    model = model.to(device)
    print(f"Number of parameters: {count_parameters(model)}")

    # < 1 increases precision, > 1 recall
    weight = torch.tensor([params["pos_weight"]], dtype=torch.float32).to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=weight)
    optimizer = torch.optim.SGD(model.parameters(), 
                                lr=params["learning_rate"],
                                momentum=params["sgd_momentum"],
                                weight_decay=params["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=params["scheduler_gamma"])
    
    # Start training
    best_loss = 1000
    early_stopping_counter = 0
    for epoch in range(300): 
        if early_stopping_counter <= 10: # = x * 5 
            # Training
            model.train()
            loss = train_one_epoch(epoch, model, train_loader, optimizer, loss_fn)
            print(f"Epoch {epoch} | Train Loss {loss}")

            # Testing
            model.eval()
            if epoch % 5 == 0:
                loss = test(epoch, model, test_loader, loss_fn)
                print(f"Epoch {epoch} | Test Loss {loss}")
                
                # Update best loss
                if float(loss) < best_loss:
                    best_loss = loss
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1

            scheduler.step()
        else:
            print("Early stopping due to no improvement.")
            return [best_loss]
    print(f"Finishing training with best test loss: {best_loss}")
    return [best_loss]

# Hyperparameter search
print("Running hyperparameter search...")
hyperparameters = {
    "learning_rate": [0.001, 0.01, 0.1],
    "batch_size": [32, 64, 128],
    "sgd_momentum": [0.9, 0.95, 0.99],
    "weight_decay": [0.001, 0.01, 0.1],
    "scheduler_gamma": [0.1, 0.5, 0.9],
    "pos_weight": [0.5, 1.0, 2.0],
    "batch_size": [32, 128, 64],
    "learning_rate": [0.1, 0.05, 0.01, 0.001],
    "weight_decay": [0.0001, 0.00001, 0.001],
    "sgd_momentum": [0.9, 0.8, 0.5],
    "scheduler_gamma": [0.995, 0.9, 0.8, 0.5, 1],
    "pos_weight" : [1.0],  
    "model_embedding_size": [8, 16, 32, 64, 128],
    "model_attention_heads": [1, 2, 3, 4],
    "model_layers": [3],
    "model_dropout_rate": [0.2, 0.5, 0.9],
    "model_top_k_ratio": [0.2, 0.5, 0.8, 0.9],
    "model_top_k_every_n": [0],
    "model_dense_neurons": [16, 128, 64, 256, 32]
}

best_params = None
best_loss = float('inf')

for _ in range(100):  # Number of iterations for the search
    params = {key: random.choice(values) for key, values in hyperparameters.items()}
    loss = run_one_training(params)
    if loss < best_loss:
        best_loss = loss
        best_params = params

print("Best hyperparameters:", best_params)
print("Best test loss:", best_loss)
