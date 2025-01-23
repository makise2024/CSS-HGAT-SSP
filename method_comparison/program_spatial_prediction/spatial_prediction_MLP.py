'''Calculate the experimental results of the semantic-based method MLP.'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, roc_auc
from sklearn.preprocessing import label_binarize

# Load the node attributes file.
nodes_df = pd.read_csv('../experimental_data/Grid500_AllCity_Nodes.csv')

# Filter out nodes with missing labels.
train_df = nodes_df.dropna(subset=['newLabel'])

# Convert labels to integers.
train_df['newLabel'] = train_df['newLabel'].astype(int)

# Select feature and label columns.
drop_field = ['ORIG_FID',"Flood_Sum","rain0715","rain0716","rain0717","rain0718","rain0719","rain0720","rain0721","rain0722",
                  "rain0723","rain0724","rain0725","Join_Count","FloodArea","Label",'newLabel']
X = train_df.drop(columns=drop_field).values
Y = train_df['newLabel'].values


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
X_test = test_df.drop(columns=drop_field).values

# Standardize the data.
scaler = StandardScaler()
X_train = scaler.fit_transform(X)
X_test = scaler.transform(X_test)

# Train the MLP model.
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
mlp.fit(X_train, Y)


# Make predictions.
y_pred = svm.predict(X_test)

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