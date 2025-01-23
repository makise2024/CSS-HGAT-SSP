'''Obtain unit representations for each day.'''

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


class MultiLayerGATWithChannelAttention(nn.Module):
    """
        A Multi-Layer Graph Attention Network (GAT) with Channel Attention mechanism.
        This model performs node-level representation learning using GAT layers,
        and applies channel attention to combine multi-layer outputs adaptively.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, heads=8, dropout=0.6):
        """
            Initializes the MultiLayerGATWithChannelAttention model.

            Args:
                in_channels (int): The number of input features for each node.
                hidden_channels (int): The number of hidden features for each GAT layer.
                out_channels (int): The number of output features for each node (e.g., class count in classification).
                num_layers (int): The number of GAT layers in the model.
                heads (int, optional): Number of attention heads in each GAT layer. Default is 8.
                dropout (float, optional): Dropout rate applied after each GAT layer. Default is 0.6.
        """
        super(MultiLayerGATWithChannelAttention, self).__init__()

        self.num_layers = num_layers  # Number of GAT layers
        self.heads = heads  # Number of attention heads
        self.dropout = dropout  # Dropout rate

        # Create a list of GAT layers
        self.convs = nn.ModuleList()

        # First GAT layer: input -> hidden
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads))

        # Additional GAT layers: hidden -> hidden
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads))

        # Channel attention layers: one for each GAT layer
        self.channel_attention = nn.ModuleList()
        for _ in range(num_layers):
            self.channel_attention.append(nn.Linear(hidden_channels * heads, 1))  # Linear transformation for attention

        # Fully connected layer for final output
        self.fc = nn.Linear(hidden_channels * heads, out_channels)

    def forward(self, data):
        """
           Forward pass of the model.

           Args:
               data (torch_geometric.data.Data): Input graph data containing:
                   - `x` (torch.Tensor): Node feature matrix of shape [num_nodes, in_channels].
                   - `edge_index` (torch.Tensor): Edge index tensor defining graph connectivity.

           Returns:
               torch.Tensor: Log-probabilities of shape [num_nodes, out_channels].
        """

        x, edge_index = data.x, data.edge_index
        layer_outputs = []

        # Pass through each GAT layer
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            layer_outputs.append(x)

        # Apply channel attention to combine layer outputs
        attention_weights = [torch.sigmoid(att(layer_output)) for att, layer_output in zip(self.channel_attention, layer_outputs)]
        attention_weights = torch.stack(attention_weights, dim=1)  # shape: [num_nodes, num_layers, hidden_channels * heads]
        layer_outputs = torch.stack(layer_outputs, dim=1)  # shape: [num_nodes, num_layers, hidden_channels * heads]

        # Element-wise multiplication of layer outputs and attention weights
        x = layer_outputs * attention_weights  # shape: [num_nodes, num_layers, hidden_channels * heads]
        x = torch.sum(x, dim=1)  # shape: [num_nodes, hidden_channels * heads]

        self.node_embeddings = x
        x = self.fc(x)

        # Return log-softmax probabilities for classification
        return F.log_softmax(x, dim=1)


def normalize_data(data):
    """
        Standardizes the input data to have a mean of 0 and a standard deviation of 1 for each feature.

        Args:
            data (torch.Tensor): A 2D tensor of shape [num_samples, num_features].

        Returns:
            torch.Tensor: A normalized tensor of the same shape as the input.
    """

    # Compute the mean and standard deviation for each feature
    mean = data.mean(dim=0)
    std = data.std(dim=0)

    # Avoid division by zero by setting zero standard deviations to 1
    std[std == 0] = 1

    # Normalize data: subtract the mean and divide by the standard deviation
    return (data - mean) / std


def get_node_embeddings_for_day(node_data_path, edge_data_path, model, day):
    """
        Generates node embeddings for a specific day using the given model.

        Args:
            node_data_path (str): Path to the CSV file containing node data.
            edge_data_path (str): Path to the CSV file containing edge data.
            model (torch.nn.Module): A trained GNN model to compute node embeddings.
            day (int): The specific day (e.g., 15 for the 15th) to extract data.

        Returns:
            torch.Tensor: A tensor containing node embeddings for the graph.
        """

    # Load node and edge data.
    nodes_df = pd.read_csv(node_data_path)
    edges_df = pd.read_csv(edge_data_path)

    # Define the column for the specific day.
    day_column = f'rain07{day:02d}'

    G = nx.Graph()

    # Add nodes with features and labels
    for _, row in nodes_df.iterrows():
        node_id = int(row['ORIG_FID'])
        drop_field = ['ORIG_FID',"Flood_Sum","rain0715","rain0716","rain0717","rain0718","rain0719","rain0720","rain0721","rain0722",
                      "rain0723","rain0724","rain0725","Join_Count","FloodArea","Label",'newLabel']
        row["HenanPr"] = row[day_column]
        features = row.drop(drop_field).astype(float).values
        label = row['newLabel']
        if label in [0, 1, 2]:
            label = int(label)
        else:
            label = -1
        G.add_node(node_id, x=torch.tensor(features, dtype=torch.float), y=label)

    # Add edges to the graph.
    for _, row in edges_df.iterrows():
        if int(row['TARGET_FID']) - int(row['JOIN_FID']):
            G.add_edge(int(row['TARGET_FID']), int(row['JOIN_FID']))

    # Convert the graph to PyTorch Geometric format.
    data = from_networkx(G)
    data.x = torch.stack([G.nodes[node]['x'] for node in G.nodes()])
    data.x = normalize_data(data.x)

    # Compute node embeddings using the model
    model.eval()
    with torch.no_grad():
        model(data)
        embeddings = model.node_embeddings

    # Return the computed node embeddings
    return embeddings


# Paths to the node and edge data files.
node_data_path = './experimental_data/Grid500_AllCity_Nodes.csv'
edge_data_path = './experimental_data/Grid500_AllCity_Edges.csv'

# Initialize the model.
model = MultiLayerGATWithChannelAttention(13, 16, 3, 7, heads=8, dropout=0.6)

# Generate node embeddings for each day and save them to CSV files.
days = [15,16,17,18,19,20,21,22,23,24,25]
for day in days:
    # Generate node embeddings for the specific day
    embeddings = get_node_embeddings_for_day(node_data_path, edge_data_path, model, day)

    # Save the embeddings to a CSV file
    embeddings_df = pd.DataFrame(embeddings)
    file_path = './experimental_data/(4)_temporal_embeddings/' + f'embeddings_day_{day}.csv'
    embeddings_df.to_csv(file_path)
