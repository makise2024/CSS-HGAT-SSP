'''Calculate the experimental results of the structure-based method PageRank.'''

import pandas as pd
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the node attributes file and the node edges file.
nodes_df = pd.read_csv('../experimental_data/Grid500_AllCity_Nodes.csv')
edges_df = pd.read_csv('../experimental_data/Grid500_AllCity_Edges.csv')

# Construct the graph.
G = nx.Graph()
for _, row in edges_df.iterrows():
    if int(row['TARGET_FID']) - int(row['JOIN_FID']):
        G.add_edge(int(row['TARGET_FID']), int(row['JOIN_FID']))

# Calculate PageRank values and add them to node attributes.
pagerank_scores = nx.pagerank(G)
nodes_df['pagerank'] = nodes_df['ORIG_FID'].map(pagerank_scores)

degree_centrality = nx.degree_centrality(G)
nodes_df['degree'] = nodes_df['ORIG_FID'].map(degree_centrality)

# Select feature and label columns.
feature_columns = ['pagerank', 'degree']
labeled_nodes = nodes_df.dropna(subset=['newLabel'])
X_train = labeled_nodes[feature_columns].values
y_train = labeled_nodes['newLabel'].astype(int).values

# Build the list of true labels.
true_labels = []

## Load geographic unit IDs covered by the flood monitoring image.
nodes_selected = pd.read_csv('../experimental_data/Grid500_Selected_FloodArea.csv')
all_selected = nodes_selected["ORIG_FID"]

## Load geographic unit IDs used for training.
nodes_train = pd.read_csv('../experimental_data/Grid500_AllCity_Nodes.csv')
orgin_indicate = nodes_train['Flood_Sum']
train_nodes = nodes_train.loc[nodes_train['newLabel'].notna(), 'ORIG_FID'].values

## Create labels for the test set.
all_test = [item for item in all_selected if item not in train_nodes]

for node_id in all_test:
    if orgin_indicate.loc[nodes_train['ORIG_FID'] == node_id].values[0] <= 50:
        true_labels.append(0)
    elif orgin_indicate.loc[nodes_train['ORIG_FID'] == node_id].values[0] <= 500:
        true_labels.append(1)
    else:
        true_labels.append(2)
test_df = nodes_df.loc[all_test]
X_test = test_df[feature_columns].values

# Standardize the data.
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model and make predictions.
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Compute evaluation metrics.
## Compute Weighted Precision.
weighted_precision = precision_score(true_labels, y_pred, average='weighted')
print(f'Weighted Precision: {macro_precision:.4f}')

## Compute Weighted Recall.
weighted_recall = recall_score(true_labels, y_pred, average='weighted')
print(f'Weighted Recall: {macro_recall:.4f}')

## Compute Micro ROC-AUC.
true_labels_binary = label_binarize(true_labels, classes=[0, 1, 2])
predicted_labels_binary = label_binarize(y_pred, classes=[0, 1, 2])
roc_auc_micro = roc_auc_score(true_labels_binary, predicted_labels_binary, average='micro')
print(f'Micro ROC-AUC: {roc_auc_micro:.4f}')

