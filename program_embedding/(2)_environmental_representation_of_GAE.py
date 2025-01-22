'''Calculating the environmental representation of grid cells based on the GAE method.'''

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GAE
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx, train_test_split_edges
import networkx as nx
import pandas as pd
import numpy as np

# Set the random seed to 42.
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Read data.
nodes_df = pd.read_csv('../experimental_data/Grid500_AllCity_Nodes.csv')
edges_df = pd.read_csv('../experimental_data/Grid500_AllCity_Edges.csv')

# Create a graph.
G = nx.Graph()
# Add edges (TARGET_FID, JOIN_FID) to the graph.
for _, row in nodes_df.iterrows():
    node_id = int(row['ORIG_FID'])
    G.add_node(node_id)
for _, row in edges_df.iterrows():
    if int(row['TARGET_FID']) - int(row['JOIN_FID']):
        G.add_edge(int(row['TARGET_FID']), int(row['JOIN_FID']))

# Convert to PyTorch Geometric data format.
data = from_networkx(G)
data.x = torch.eye(len(G.nodes()))

# 划分训练集和测试集
data = train_test_split_edges(data)

# 定义 GAE 模型
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)

# 模型参数
in_channels = data.x.size(1)
out_channels = 128

# Define the model and optimizer.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GAE(GCNEncoder(in_channels, out_channels)).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train():
    # Perform a training step for the graph autoencoder, including forward pass, loss computation, and backpropagation.
    model.train()
    optimizer.zero_grad()
    z = model.encode(data.x, data.train_pos_edge_index)
    loss = model.recon_loss(z, data.train_pos_edge_index)
    loss.backward()
    optimizer.step()
    return loss.item()

# Training process.
epochs = 100
for epoch in range(epochs):
    loss = train()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch + 1}, Loss: {loss:.4f}')

# Output the final node representations.
model.eval()
with torch.no_grad():
    z = model.encode(data.x, data.train_pos_edge_index)
node_embeddings = z.cpu().numpy()

node_ids = list(G.nodes())
embeddings_df = pd.DataFrame(node_embeddings, index=node_ids)
embeddings_df.to_csv('../experimental_results/(2)_embeddings/GAE_embeddings.csv')
