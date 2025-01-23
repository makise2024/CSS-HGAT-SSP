'''Calculating the environmental representation of grid cells based on the GCN method.'''

import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
import pandas as pd
import networkx as nx
import numpy as np
import random
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

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
# Add nodes and their features to the graph.
for _, row in nodes_df.iterrows():
    node_id = int(row['ORIG_FID'])
    ## Retain static feature fields for embedding.
    drop_field = ['ORIG_FID', "Flood_Sum", "rain0715", "rain0716", "rain0717", "rain0718", "rain0719", "rain0720",
                  "rain0721", "rain0722",
                  "rain0723", "rain0724", "rain0725", "Join_Count", "FloodArea", "Label", 'newLabel']
    features = row.drop(drop_field).astype(float).values
    label = row['newLabel']
    if label in [0, 1, 2]:
        label = int(label)
    else:
        # Set the labels of unlabeled units to -1.
        label = -1
    G.add_node(node_id, x=torch.tensor(features, dtype=torch.float), y=label)
# Add edges (TARGET_FID, JOIN_FID) to the graph.
for _, row in edges_df.iterrows():
    if int(row['TARGET_FID']) - int(row['JOIN_FID']):
        G.add_edge(int(row['TARGET_FID']), int(row['JOIN_FID']))

# Convert to PyTorch Geometric data format.
data = from_networkx(G)
data.x = torch.stack([G.nodes[node]['x'] for node in G.nodes()])
data.y = torch.tensor([G.nodes[node]['y'] for node in G.nodes()], dtype=torch.long)

# Create a training mask.
data.train_mask = data.y != -1
data.y[data.y == -1] = 0


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# Initialize the model.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(data.num_node_features, 128, 128).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001,weight_decay=0.01)
criterion = torch.nn.CrossEntropyLoss()
data = data.to(device)

def train():
    # Perform a single training step, including forward pass, loss computation, and backpropagation.
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

# Training process.
for epoch in range(50):
    loss = train()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch + 1}, Loss: {loss:.4f}')

# Output the final node representations.
model.eval()
with torch.no_grad():
    embeddings = model(data).cpu().numpy()

node_ids = list(G.nodes())
embeddings_df = pd.DataFrame(embeddings, index=node_ids)
embeddings_df.to_csv('../experimental_results/(2)_embeddings/GCN_embeddings.csv')
