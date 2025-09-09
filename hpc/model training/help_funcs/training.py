#Torch
import torch

#Other
from sklearn.metrics import f1_score

def val_loop(model, val_loader, device, loss_func, loss_name):
    model.eval()
    val_loss = 0
    all_predictions = []
    all_labels = []
    for batch in val_loader:
        x, edge_index, edge_weight, data = batch.x.to(device), batch.edge_index.to(device), batch.edge_attr.to(device), batch.batch.to(device)
        #batch = batch.to(device, non_blocking=True)
        y_hat = model.forward(x, edge_index, edge_weight, data)
        
        y = batch.y
        if loss_name == 'BCE':
            y = y.float()
            y_hat = y_hat.squeeze(1)
        y = y.to(device)

        loss = loss_func(y_hat, y)
        val_loss += loss

        if loss_name == 'BCE':
            all_predictions += y_hat.round().tolist()
        else:
            all_predictions += torch.argmax(y_hat, dim=-1).tolist()
        all_labels += batch.y.tolist()

    return val_loss/len(val_loader), f1_score(all_labels, all_predictions, average='micro'), [all_labels, all_predictions]

def train_loop(model, train_loader, optimizer, device, loss_func, loss_name):
    model.train()
    train_loss = 0.0
    for batch in train_loader:
        #batch = batch.to(device, non_blocking=True)
        x, edge_index, edge_weight, data = batch.x.to(device), batch.edge_index.to(device), batch.edge_attr.to(device), batch.batch.to(device)

        optimizer.zero_grad()
        out = model(x, edge_index, edge_weight, data)
        
        y = batch.y
        if loss_name == 'BCE':
            y = y.float()
            out = out.squeeze(1)
        y = y.to(device)
        loss = loss_func(out, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    return train_loss / len(train_loader)