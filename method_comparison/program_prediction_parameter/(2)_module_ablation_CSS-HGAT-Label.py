'''CSS-HGAT-Label method (Module ablation experiment of disaster risk prediction)'''

import pandas as pd
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score)
from sklearn.preprocessing import label_binarize

# Step 1: Read node representations and label data.
df = pd.read_csv('./experimental_results/4.2.1_neighborhood_selection/GAT_embeddings_predictions_7.csv', index_col=0)
embeddings = df.drop(columns=['predicted_label']).values
node_ids = df.index.tolist()

## Read the geographic unit numbers used for training
nodes_train = pd.read_csv('./experimental_data/Grid500_AllCity_Nodes.csv')

# Step 2: Compute evaluation metrics.
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
    predicted_labels.append(preds[node_id])

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
