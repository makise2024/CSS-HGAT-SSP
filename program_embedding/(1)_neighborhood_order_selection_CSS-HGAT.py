'''Evaluate the effect of aggregating representations from different orders of neighborhoods.'''

import pandas as pd
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

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
    drop_field = ['ORIG_FID',"Flood_Sum","rain0715","rain0716","rain0717","rain0718","rain0719","rain0720","rain0721","rain0722",
                  "rain0723","rain0724","rain0725","Join_Count","FloodArea","Label",'newLabel']
    features = row.drop(drop_field).astype(float).values
    label = row['newLabel']
    if label in [0,1,2]:
        label = int(label)
    else:
	## Set the labels of unlabeled units to -1.
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

# Feature normalization.
data.x = (data.x - data.x.mean(dim=0)) / data.x.std(dim=0)

# Create a training mask.
data.train_mask = data.y != -1
data.y[data.y == -1] = 0  


class MultiLayerGATWithChannelAttention(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, heads=8, dropout=0.6):
        super(MultiLayerGATWithChannelAttention, self).__init__()
        self.num_layers = num_layers
        self.heads = heads
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads))
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads))

        self.channel_attention = nn.ModuleList()
        for _ in range(num_layers):
            self.channel_attention.append(nn.Linear(hidden_channels * heads, 1))

        self.fc = nn.Linear(hidden_channels * heads, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        layer_outputs = []

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            layer_outputs.append(x)

        # Apply channel attention.
        attention_weights = [torch.sigmoid(att(layer_output)) for att, layer_output in zip(self.channel_attention, layer_outputs)]
        attention_weights = torch.stack(attention_weights, dim=1)  # shape: [num_nodes, num_layers, hidden_channels * heads]
        layer_outputs = torch.stack(layer_outputs, dim=1)  # shape: [num_nodes, num_layers, hidden_channels * heads]

        x = layer_outputs * attention_weights  # shape: [num_nodes, num_layers, hidden_channels * heads]
        x = torch.sum(x, dim=1)  # shape: [num_nodes, hidden_channels * heads]

        self.node_embeddings = x  # Save the final node embeddings

        x = self.fc(x)
        return F.log_softmax(x, dim=1)

for num_layers in range(1,10):
    # Define the model and optimizer.
    model = MultiLayerGATWithChannelAttention(data.num_node_features, 16, 3, num_layers, heads=8, dropout=0.6)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-5)

    # Loss function.
    def label_smoothing_loss(output, target, smoothing=0.1):
        confidence = 1.0 - smoothing
        log_probs = F.log_softmax(output, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = confidence * nll_loss + smoothing * smooth_loss
        return loss.mean()

    def train():
        # Perform a single training step, including forward pass, loss computation, and backpropagation.
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = label_smoothing_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        return loss.item()

    def test():
        # Evaluate the model's accuracy on the training dataset.
        model.eval()
        out = model(data)
        pred = out.argmax(dim=1)
        correct = (pred[data.train_mask] == data.y[data.train_mask]).sum()
        acc = int(correct) / int(data.train_mask.sum())
        return acc

    # Training process.
    for epoch in range(100):
        loss = train()
        accuracy = test()
        print(f'Epoch {epoch + 1}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}')

    # Save the model weights.
    torch.save(model.state_dict(), 'gat_model-7_0707.pth')

    # Predict the labels of all nodes.
    model.eval()
    out = model(data)
    pred = out.argmax(dim=1).detach().numpy()

    # Output the final node representations.
    node_embeddings = model.node_embeddings.detach().numpy()

    node_ids = list(G.nodes())
    results_df = pd.DataFrame(node_embeddings, index=node_ids)
    results_df['predicted_label'] = pred
    file_name = '../experimental_results/(1)_neighborhood_selection/' + f'GAT_embeddings_predictions_{num_layers}.csv'
    results_df.to_csv(file_name)