import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GATConv
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import numpy as np

cora_dataset = Planetoid('/tmp/cora', 'cora')
cora_data = cora_dataset[0]

# Define the GAT model
class GAT(torch.nn.Module):
    # hidden channels will be the embedding dimension for each attention head
    # after applying the first GAT layer.
    def __init__(self, in_channels, hidden_channels, 
                 num_heads, dropout_rate, num_classes):
        super().__init__()
        
        self.dropout_rate = dropout_rate
        
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        
        self.conv1 = GATConv(in_channels, hidden_channels, heads=num_heads, 
                                dropout=dropout_rate)
        self.conv2 = GATConv(hidden_channels*num_heads, num_classes, 
                                dropout=dropout_rate, concat=False)

    def forward(self, x, edge_index):
        out = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        out = self.conv1(out, edge_index)
        assert out.shape[-1] == self.hidden_channels * self.num_heads
        
        out = F.elu(out)
        out = F.dropout(out, p=self.dropout_rate, training=self.training)
        
        out = self.conv2(out, edge_index)
        return out


def train(model, data, optimizer, loss_fn):
    model.train()
    optimizer.zero_grad()
    
    pred = model(data.x, data.edge_index)
    loss = loss_fn(pred[data.train_mask], data.y[data.train_mask])
    
    loss.backward()
    optimizer.step()
    
    return loss


@torch.no_grad()
def evaluate(model, data, test_mask, loss_fn):
    accuracy_list = [0.0, 0.0]
    loss_list = [0.0, 0.0]
    model.eval()

    logits = model(data.x, data.edge_index)
    pred = logits.argmax(dim=-1)
    
    for i, mask in enumerate([data.train_mask, test_mask]):
        accuracy_list[i] = pred[mask].eq(data.y[mask]).float().mean().item()
        loss_list[i] = loss_fn(logits[mask], data.y[mask]).item()

    return accuracy_list, loss_list


def train_model(model, data, optimizer, loss_fn, save_name, num_epochs=1000, 
                log_freq=50, patience=100):
    print(f"Using {device}\n")
    print(f"model devide: {next(model.parameters()).device}")
    
    # Early stopping initialization
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0
    best_epoch = 0
    best_model_state = None
    
    # Evaluate before training
    acc_list, loss_list = evaluate(model, data, data.val_mask, loss_fn)
    print("Before training: ")
    print(f"Train Acc: {acc_list[0]:.4f}, Train Loss: {loss_list[0]:.4f}, Val Acc: {acc_list[1]:.4f}, Val Loss: {loss_list[1]:.4f}\n")
    
    # Start training
    for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
        loss = train(model, data, optimizer, loss_fn)
        acc_list, loss_list = evaluate(model, data, 
                                       data.val_mask, loss_fn)
    
        # Update early stopping criteria
        val_loss = loss_list[1]
        val_acc = acc_list[1]
        if val_loss < best_val_loss or val_acc > best_val_acc:
            best_val_loss = min(best_val_loss, val_loss)
            best_val_acc = max(best_val_acc, val_acc)
            patience_counter = 0
            best_epoch = epoch
            torch.save(model.state_dict(), save_name)
        else:
            patience_counter += 1
    
        # Check if patience limit is reached
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch + 1}")
            break
            
        # Logging
        if (epoch % log_freq == 0) or (epoch + 1 == num_epochs):
            print(f"Epoch: {epoch+1}, Loss: {loss:.4f}")
            print(f"    Eval: Train Acc: {acc_list[0]:.4f}, Train Loss: {loss_list[0]:.4f}, Val Acc: {acc_list[1]:.4f}, Val Loss: {loss_list[1]:.4f}")

    if best_model_state is not None:
        print(f"model devide: {next(model.parameters()).device}")
        model.load_state_dict(best_model_state["state_dict"])
        print(f"Model restored to the best state from epoch {best_epoch + 1}")
        print(f"model devide: {next(model.parameters()).device}")
        
    print(f"\nTraining completed.\nBest Validation at Epoch: {best_epoch + 1}\nBest Val Acc: {best_val_acc:.4f}, Best Val Loss: {best_val_loss:.4f}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_heads = 8
dropout_rate = 0.4
emb_dim1 = 8
lr = 0.005

cora_num_classes = len(cora_data.y.unique())
assert cora_num_classes == 7


num_epochs = 1000
log_freq = 50

cora_model = GAT(cora_data.num_features, emb_dim1, num_heads, dropout_rate, 
            cora_num_classes).to(device)
cora_data.to(device)

lambda_l2 = 0.001
cora_optimizer = torch.optim.Adam(cora_model.parameters(), lr=lr, weight_decay=lambda_l2)
loss_fn = nn.CrossEntropyLoss()


train_model(cora_model, cora_data, cora_optimizer, loss_fn, "cora_model_01.pth", num_epochs, log_freq, patience=100)
cora_model.load_state_dict(torch.load("cora_model_01.pth", map_location=device))
cora_model = cora_model.to(device)


# Evaluate after training
acc_list, loss_list = evaluate(cora_model, cora_data, cora_data.val_mask, loss_fn)
print("After training: ")
print(f"Train Acc: {acc_list[0]:.4f}, Train Loss: {loss_list[0]:.4f}, Val Acc: {acc_list[1]:.4f}, Val Loss: {loss_list[1]:.4f}\n")


cite_dataset = Planetoid('/tmp/Citeseer', 'Citeseer')
cite_data = cite_dataset[0]


cite_num_classes = len(cite_data.y.unique())
assert cite_num_classes == 6

cite_model = GAT(cite_data.num_features, emb_dim1, num_heads, dropout_rate, 
            cite_num_classes).to(device)
cite_data.to(device)

# Create optimizer for citeseer
cite_optimizer = torch.optim.Adam(cite_model.parameters(), lr=lr, weight_decay=lambda_l2)

# Train citeseer model
train_model(cite_model, cite_data, cite_optimizer, loss_fn, "cite_model_01.pth", num_epochs=1000, log_freq=10, patience=100)
cite_model.load_state_dict(torch.load("cite_model_01.pth", map_location=device))
cite_model = cite_model.to(device)

# Evaluate after training
acc_list, loss_list = evaluate(cite_model, cite_data, cite_data.val_mask, loss_fn)
print("\nAfter training: ")
print(f"Train Acc: {acc_list[0]:.4f}, Train Loss: {loss_list[0]:.4f}, Val Acc: {acc_list[1]:.4f}, Val Loss: {loss_list[1]:.4f}\n")
