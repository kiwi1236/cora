import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GATConv
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import numpy as np


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


def summarize(model):
    num_params = 0
    print(f"Model Summary: {type(model).__name__}\n")
    for name, param in model.named_parameters():
        print(name, param.size())
        num_params += param.numel()
    print(f"\nTotal number of params: {num_params}")


cora_dataset = Planetoid('/tmp/cora', 'cora')
cora_data = cora_dataset[0]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_heads = 8
dropout_rate = 0.4
emb_dim1 = 8
lr = 0.005

cora_num_classes = len(cora_data.y.unique())
assert cora_num_classes == 7

num_epochs = 100
log_freq = 10

cora_model = GAT(cora_data.num_features, emb_dim1, num_heads, dropout_rate, 
            cora_num_classes).to(device)
cora_data.to(device)

lambda_l2 = 0.001
optimizer = torch.optim.Adam(cora_model.parameters(), lr=lr, weight_decay=lambda_l2)
loss_fn = nn.CrossEntropyLoss()

print(f"Using {device}\n")

# Evaluate before training
acc_list, loss_list = evaluate(cora_model, cora_data, cora_data.val_mask, loss_fn)
print("Before training: ")
print(f"Train Acc: {acc_list[0]:.4f}, Train Loss: {loss_list[0]:.4f}, Val Acc: {acc_list[1]:.4f}, Val Loss: {loss_list[1]:.4f}\n")

# Start training
for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
    loss = train(cora_model, cora_data, optimizer, loss_fn)
    acc_list, loss_list = evaluate(cora_model, cora_data, cora_data.val_mask, loss_fn)
    
    if (epoch % log_freq == 0) or (epoch + 1 == num_epochs):
        print(f"Epoch: {epoch+1}, Loss: {loss:.4f}")
        print(f"           Train Acc: {acc_list[0]:.4f}, Train Loss: {loss_list[0]:.4f}, Val Acc: {acc_list[1]:.4f}, Val Loss: {loss_list[1]:.4f}")

