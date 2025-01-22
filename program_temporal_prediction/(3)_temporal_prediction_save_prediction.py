'''Predict the disaster risk for July 21 based on the representations from the previous four days and save the results to a file.'''

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


def load_model(model, optimizer, checkpoint_path):
    """
    Loads a model and optimizer state from a checkpoint file.

    Args:
        model (torch.nn.Module): The model to load the state into.
        optimizer (torch.optim.Optimizer): The optimizer to load the state into.
        checkpoint_path (str): Path to the checkpoint file.

    Returns:
        tuple: The epoch number and accuracy stored in the checkpoint.
    """
    # Load checkpoint file.
    checkpoint = torch.load(checkpoint_path)

    # Restore model and optimizer states.
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Return saved epoch and accuracy.
    return checkpoint['epoch'], checkpoint['accuracy']

def predict(model, embeddings_tensor):
    """
    Generates predictions using a trained model on the provided embeddings.

    Args:
        model (torch.nn.Module): The trained model to use for prediction.
        embeddings_tensor (torch.Tensor): Input tensor of node embeddings.

    Returns:
        torch.Tensor: Predicted class labels for each node.
    """
    model.eval()  # Set model to evaluation mode (disables dropout, batch normalization).

    with torch.no_grad():  # Disable gradient computation for faster inference.
        outputs = model(embeddings_tensor)  # Perform forward pass to compute outputs.
        _, preds = torch.max(outputs, 1)  # Extract predicted class indices.

    return preds  # Return predictions as a tensor.


# Predicting the risk of the fifth day based on the representations of the previous four days。
k_days = [17,18,19,20]
embeddings_list = []
for day in k_days:
    file_path = '../experimental_results/(4)_temporal_embeddings/' + f'embeddings_day_{day}.csv'
    embeddings = load_embeddings_from_csv(file_path)

    # Construct adjacency matrix and convert to a graph using k-nearest neighbors.
    adj_matrix = kneighbors_graph(embeddings, 4, mode='connectivity', include_self=True)
    G = nx.from_scipy_sparse_array(adj_matrix)
    print(f"Number of nodes in graph G: {G.number_of_nodes()}")

    # Extract graph-based neighborhood features: degree and clustering coefficient.
    degree = np.array([val for (node, val) in G.degree()])
    clustering_coefficient = np.array(list(nx.clustering(G).values()))

    # Normalize neighborhood features for consistent scaling.
    scaler = StandardScaler()
    degree_scaled = scaler.fit_transform(degree.reshape(-1, 1)).flatten()
    clustering_coefficient_scaled = scaler.fit_transform(clustering_coefficient.reshape(-1, 1)).flatten()

    # Combine original embeddings with normalized neighborhood features.
    combined_features = np.hstack((embeddings, degree_scaled.reshape(-1, 1), clustering_coefficient_scaled.reshape(-1, 1)))

    embeddings_list.append(torch.tensor(np.expand_dims(combined_features, axis=0), dtype=torch.float))

# Combine embeddings into (num_nodes, k, num_features).
test_embeddings_tensor = torch.cat(embeddings_list, dim=0).permute(1, 0, 2)  # (num_nodes, k, num_features)

# Define model parameters.
input_dim = test_embeddings_tensor.size(2)
hidden_dim = 64
output_dim = 3
num_heads = 5
num_layers = 2

# Initialize model, loss function, and optimizer.
model = TemporalTransformerModel(input_dim, hidden_dim, output_dim, num_heads, num_layers)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Load model.
checkpoint_path = '../experimental_results/(5)_temporal_model/CSS-HGAT-SSP_best_model_4_days.pth'
epoch, accuracy = load_model(model, optimizer, checkpoint_path)
print(f'Loaded model from epoch {epoch} with accuracy {accuracy:.4f}')

# Predicting the risk of the fifth day and save the result.
predictions = predict(model, test_embeddings_tensor)

output_path = '../experimental_results/(6)_temporal_prediction/predictions_21.csv' #Used to draw Fig.15 and Fig.16.
predictions_np = predictions.cpu().numpy()
df_predictions = pd.DataFrame(predictions_np, columns=['Prediction'])
df_predictions.to_csv(output_path, index=False)
print(f'Predictions saved to {output_path}')