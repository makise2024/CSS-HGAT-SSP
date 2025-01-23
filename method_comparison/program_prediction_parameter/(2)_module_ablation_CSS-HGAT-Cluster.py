'''CSS-HGAT-Cluster method (Module ablation experiment of disaster risk prediction)'''

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
from sklearn.metrics import (precision_score, recall_score, roc_auc_score)
from sklearn.preprocessing import label_binarize


# Step 1: Read node representations and label data.
df = pd.read_csv('./experimental_results/4.2.1_neighborhood_selection/GAT_embeddings_predictions_7.csv', index_col=0)
embeddings = df.drop(columns=['predicted_label']).values
node_ids = df.index.tolist()

## Read the geographic unit numbers used for training
nodes_train = pd.read_csv('./experimental_data/Grid500_AllCity_Nodes.csv')

## Filter out entries where the newLabel field is empty and ensure its values are of type int.
node_labels = {row['ORIG_FID']: int(row['newLabel']) for _, row in nodes_train.iterrows() if pd.notnull(row['newLabel'])}


# Step 2: Clustering
num_clusters = len(set(node_labels.values()))
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
kmeans.fit(embeddings)
cluster_labels = kmeans.labels_


# Step 3: Label propagation.
## Create a dictionary to store the final node labels
final_node_labels = {}

## Add the labels of nodes with existing labels to final_node_labels.
for node, label in node_labels.items():
    final_node_labels[node] = label

## Create a dictionary to store the label distribution for each cluster.
cluster_label_count = {i: Counter() for i in range(num_clusters)}

## Iterate through all nodes and update the label distribution for each cluster based on existing labels.
for i, node_id in enumerate(node_ids):
    if node_id in node_labels:
        cluster_label_count[cluster_labels[i]][node_labels[node_id]] += 1
# print("Cluster label distribution:")
# for cluster_id, label_count in cluster_label_count.items():
#     print(f"Cluster {cluster_id}: {label_count}")

## Perform label propagation for unlabeled nodes
for i, node_id in enumerate(node_ids):
    if node_id not in node_labels:
        most_common_label = cluster_label_count[cluster_labels[i]].most_common(1)
        if most_common_label:
            final_node_labels[node_id] = most_common_label[0][0]
# print("Final node labels:", final_node_labels)


# Step 4: Compute evaluation metrics.
## Load geographic unit IDs covered by the flood monitoring image.
nodes_selected = pd.read_csv('./experimental_data/Grid500_Selected_FloodArea.csv')
all_selected = nodes_selected["ORIG_FID"]

## Load geographic unit IDs used for training.
orgin_indicate = nodes_train['Flood_Sum']
train_nodes = nodes_train.loc[nodes_train['newLabel'].notna(), 'ORIG_FID'].values

## Read the true labels and predicted labels of the test set.
true_labels = []
predicted_labels = []
all_test = [item for item in all_selected if item not in train_nodes]
for node_id in all_test:
    if orgin_indicate.loc[nodes_train['ORIG_FID'] == node_id].values[0] <= 50:
        true_labels.append(0)
    elif orgin_indicate.loc[nodes_train['ORIG_FID'] == node_id].values[0] <= 500:
        true_labels.append(1)
    else:
        true_labels.append(2)
    predicted_labels.append(final_node_labels.get(node_id, -1))

## Convert to a numpy array.
true_labels = np.array(true_labels)
predicted_labels = np.array(predicted_labels)

## Compute Weighted Precision.
weighted_precision = precision_score(true_labels, predicted_labels, average='weighted')
print(f'Weighted Precision: {macro_precision:.4f}')

## Compute Weighted Recall.
weighted_recall = recall_score(true_labels, predicted_labels, average='weighted')
print(f'Weighted Recall: {macro_recall:.4f}')

## Compute Micro ROC-AUC.
true_labels_binary = label_binarize(true_labels, classes=[0, 1, 2])
predicted_labels_binary = label_binarize(predicted_labels, classes=[0, 1, 2])
roc_auc_micro = roc_auc_score(true_labels_binary, predicted_labels_binary, average='micro')
print(f'Micro ROC-AUC: {roc_auc_micro:.4f}')


