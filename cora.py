import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric import nn
import torch_geometric.transforms as T
import numpy as np

device = torch.device("cuda")
cora_dataset = Planetoid('/tmp/cora', 'cora')
cora_data = cora_dataset[0]

cora_x_train = cora_data.x[cora_data.train_mask]
cora_x_val = cora_data.x[cora_data.val_mask]
cora_x_test = cora_data.x[cora_data.test_mask]
