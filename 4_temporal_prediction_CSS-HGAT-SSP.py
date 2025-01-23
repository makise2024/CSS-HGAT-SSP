'''Disaster risk temporal prediction results based on CSS-HGAT-SSP (our method).'''

import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from sklearn.metrics import accuracy_score
import networkx as nx
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler


class PositionalEncoding(nn.Module):
    """
        Implements positional encoding to inject positional information into sequence embeddings.
    """
    def __init__(self, d_model, max_len=5000):
        """
           Initializes the positional encoding matrix.

           Args:
               d_model (int): Dimensionality of the embeddings.
               max_len (int): Maximum length of the sequences to be encoded.
        """
        super(PositionalEncoding, self).__init__()

        # Create positional encoding matrix using sine and cosine functions.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # Apply sine to even indices.
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply cosine to odd indices.
        pe = pe.unsqueeze(0).transpose(0, 1)  # Adjust dimensions for batch compatibility.

        # Register positional encoding matrix as a non-trainable buffer.
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
            Adds positional encoding to the input tensor.

            Args:
                x (torch.Tensor): Input tensor of shape (seq_len, batch_size, d_model).

            Returns:
                torch.Tensor: Tensor with positional encoding added.
        """
        # Add positional encoding to input tensor based on sequence length.
        return x + self.pe[:x.size(0), :]


class TemporalTransformerModel(nn.Module):
    """
       A Temporal Transformer model for sequence data.

       Attributes:
           num_layers (int): Number of transformer encoder layers.
           transformer_layers (nn.ModuleList): A list of nn.TransformerEncoderLayer instances.
           fc (nn.Linear): Fully connected layer for final classification.

       Methods:
           forward(x): Performs a forward pass of the input tensor through the model.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads, num_layers):
        """
            Initialize the TemporalTransformerModel.

            Args:
                input_dim (int): Dimensionality of the input features.
                hidden_dim (int): Dimensionality of the hidden layers (not used directly in this implementation).
                output_dim (int): Number of output classes.
                num_heads (int): Number of attention heads in the transformer encoder.
                num_layers (int): Number of transformer encoder layers.
        """
        super(TemporalTransformerModel, self).__init__()
        self.num_layers = num_layers

        # Initialize transformer encoder layers.
        self.positional_encoding = PositionalEncoding(input_dim)
        self.transformer_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.transformer_layers.append(nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads))

        # Fully connected layer for classification
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        """
            Forward pass through the model.

            Args:
                x (torch.Tensor): Input tensor of shape (num_nodes, k, num_features),
                                  where num_nodes is the number of nodes,
                                  k is the temporal dimension (number of time steps),
                                  and num_features is the feature dimension.

            Returns:
                torch.Tensor: Output tensor of shape (num_nodes, output_dim).
        """
        # Transpose tensor for transformer input (k, num_nodes, num_features).
        x = x.permute(1, 0, 2)

        # Pass through transformer layers.
        x = self.positional_encoding(x)
        for i in range(self.num_layers):
            x = self.transformer_layers[i](x)

        # Extract out for the last time step.
        x = x.permute(1, 0, 2)
        x = x[:, -1, :]

        # Pass the processed features through the final fully connected layer.
        out = self.fc(x)
        return out

def load_embeddings_from_csv(file_path):
    """
        Load node embeddings from a CSV file and convert them to a PyTorch tensor.

        Args:
            file_path (str): Path to the CSV file containing embeddings.

        Returns:
            torch.Tensor: A tensor of shape (num_nodes, num_features) with the loaded embeddings.
    """
    df = pd.read_csv(file_path)
    embeddings_np = df.values
    embeddings = torch.tensor(embeddings_np, dtype=torch.float)
    return embeddings

# Load selected nodes for labeling.
nodes_selected = pd.read_csv('./experimental_data/Grid500_Selected_FloodArea.csv')
all_selected = nodes_selected["ORIG_FID"].values

# Load data and generate labels based on flood conditions.
node_data_path = './experimental_data/Grid500_AllCity_Nodes.csv'
nodes_df = pd.read_csv(node_data_path)
labels = []

# Assign labels: 0 (low risk), 1 (moderate risk), 2 (high risk), -1 (unlabeled).
for i in range(len(nodes_df)):
    if i in all_selected:
        if nodes_df.loc[i, 'Flood_Sum'] <= 50:
            labels.append(0)
        elif nodes_df.loc[i, 'Flood_Sum'] <= 500:
            labels.append(1)
        else:
            labels.append(2)
    else:
        labels.append(-1)
        num = num + 1

labels = np.array(labels)
mask = labels != -1
train_labels = torch.tensor(labels[mask], dtype=torch.long)

# Define temporal windows for training
days = [15, 16, 17, 18, 19]

for k in range(1, 6):
    """
       Train the model using embeddings from the last k days.
    """
    # Select the last k days.
    k_days = days[-k:]
    embeddings_list = []

    # Load embeddings for the selected days.
    for day in k_days:
        file_path = './experimental_data/(4)_temporal_embeddings/' + f'embeddings_day_{day}.csv'
        embeddings = load_embeddings_from_csv(file_path)

        # Construct adjacency matrix and convert to a graph using k-nearest neighbors.
        adj_matrix = kneighbors_graph(embeddings, 4, mode='connectivity', include_self=True)
        G = nx.from_scipy_sparse_array(adj_matrix)
        # print(f"Number of nodes in graph G: {G.number_of_nodes()}")

        # Extract graph-based neighborhood features: degree and clustering coefficient.
        degree = np.array([val for (node, val) in G.degree()])
        clustering_coefficient = np.array(list(nx.clustering(G).values()))

        # Normalize neighborhood features for consistent scaling.
        scaler = StandardScaler()
        degree_scaled = scaler.fit_transform(degree.reshape(-1, 1)).flatten()
        clustering_coefficient_scaled = scaler.fit_transform(clustering_coefficient.reshape(-1, 1)).flatten()

        # Combine original embeddings with normalized neighborhood features.
        combined_features = np.hstack(
            (embeddings, degree_scaled.reshape(-1, 1), clustering_coefficient_scaled.reshape(-1, 1)))

        embeddings_list.append(torch.tensor(np.expand_dims(combined_features, axis=0),dtype=torch.float))

    # Combine embeddings into (num_nodes, k, num_features).
    embeddings_tensor = torch.cat(embeddings_list, dim=0).permute(1, 0, 2)  # (num_nodes, k, num_features)

    # Define model parameters.
    input_dim = embeddings_tensor.size(2)
    hidden_dim = 64
    output_dim = 3
    num_heads = 5
    num_layers = 2

    # Initialize model, loss function, and optimizer.
    input_dim = embeddings_tensor.size(2)
    if input_dim % num_heads != 0:
        input_dim = (input_dim // num_heads + 1) * num_heads
    temporal_transformer_model = TemporalTransformerModel(input_dim, hidden_dim, output_dim, num_heads, num_layers)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(temporal_transformer_model.parameters(), lr=0.0001)

    # Create DataLoader.
    train_labels = torch.tensor(labels[mask], dtype=torch.long)
    train_dataset = torch.utils.data.TensorDataset(embeddings_tensor[mask], train_labels)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Train the model.
    num_epochs = 100
    best_accuracy = 0.0

    for epoch in range(num_epochs):
        """
            Perform one training epoch.
        """
        temporal_transformer_model.train()
        epoch_loss = 0
        all_preds = []
        all_labels = []

        for batch_embeddings, batch_labels in train_loader:
            # Forward pass.
            optimizer.zero_grad()
            outputs = temporal_transformer_model(batch_embeddings)
            loss = criterion(outputs, batch_labels)

            # Handle potential NaN loss.
            if torch.isnan(loss):
                print(f'NaN loss detected at epoch {epoch + 1}')
                print(f'Batch embeddings: {batch_embeddings}')
                print(f'Batch labels: {batch_labels}')
                break

            # Backward pass and optimization.
            loss.backward()
            torch.nn.utils.clip_grad_norm_(temporal_transformer_model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

            # Collect predictions for accuracy computation.
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())

        # Calculate epoch accuracy.
        epoch_accuracy = accuracy_score(all_labels, all_preds)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(train_loader):.4f}, Accuracy: {epoch_accuracy:.4f}')

        # Save the best prediction model.
        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            print("the best predict result for " + f'{k}' + " days:")
            print(f'accuracy: {best_accuracy:.4f}')
            print(f'loss: {epoch_loss / len(train_loader):.4f}')

            torch.save({
                'model_state_dict': temporal_transformer_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'accuracy': best_accuracy
            }, './experimental_results/(5)_temporal_model' + f'CSS-HGAT-SSP_best_model_{k}_days.pth')
            print(f'Best model saved with accuracy: {best_accuracy:.4f}')

            # Write the best prediction to a file.
            # output_path = './experimental_data/(6)_temporal_prediction/' + f'embeddings_day_{day}.csv'
            # df_predictions = pd.DataFrame(all_preds, columns=['Prediction'])
            # df_predictions.to_csv(output_path, index=False)
            # print(f'Predictions saved to {output_path}')
