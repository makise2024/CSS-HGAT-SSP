'''Calculate the experimental results of the structure-based method DeepWalk.'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score


# Step 1: Read node representations and label data.
embeddings_df = pd.read_csv("../experimental_results./(2)_embeddings/DeepWalk_embeddings.csv", index_col=0)
node_embeddings = embeddings_df.values

## Read the geographic unit numbers used for training
nodes_df = pd.read_csv('../experimental_data/Grid500_AllCity_Nodes.csv')
embeddings_df['newLabel'] = nodes_df['newLabel']

## Filter out entries where the newLabel field is empty and ensure its values are of type int.
train_df = embeddings_df.dropna(subset=['newLabel'])
train_df['newLabel'] = train_df['newLabel'].astype(int)

## Select feature and label columns.
drop_field = ['ORIG_FID',"Flood_Sum","rain0715","rain0716","rain0717","rain0718","rain0719","rain0720","rain0721","rain0722",
                  "rain0723","rain0724","rain0725","Join_Count","FloodArea","Label",'newLabel']
X = train_df.drop(columns=['newLabel']).values
Y = train_df['newLabel'].values

## Load geographic unit IDs covered by the flood monitoring image.
nodes_selected = pd.read_csv('../experimental_data/Grid500_Selected_FloodArea.csv')
all_selected = nodes_selected["ORIG_FID"]

## Load geographic unit IDs used for training.
nodes_train = pd.read_csv('../experimental_data/Grid500_AllCity_Nodes.csv')
orgin_indicate = nodes_train['Flood_Sum']
train_nodes = nodes_train.loc[nodes_train['newLabel'].notna(), 'ORIG_FID'].values

## Create true labels for the test set.
true_labels = []
all_test = [item for item in all_selected if item not in train_nodes]
for node_id in all_test:
    if orgin_indicate.loc[nodes_train['ORIG_FID'] == node_id].values[0] <= 50:
        true_labels.append(0)
    elif orgin_indicate.loc[nodes_train['ORIG_FID'] == node_id].values[0] <= 500:
        true_labels.append(1)
    else:
        true_labels.append(2)
test_df = embeddings_df.loc[all_test]
X_test = test_df.drop(columns=["newLabel"]).values

# Step 2: Training and prediction.
## Standardize data.
scaler = StandardScaler()
X_train = scaler.fit_transform(X)
X_test = scaler.transform(X_test)

## Classify based on the vector obtained by the DeepWalk model.
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, Y)
y_pred = model.predict(X_test)


# Step 3: Compute evaluation metrics.
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

