'''Calculate the experimental results of methods that combine semantic and structural information (GCN, GraphSAGE, and our proposed CSS-HGAT-SSP).'''

import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import label_binarize

file_list =['GCN_embeddings.csv','GraphSAGE_embeddings.csv','GAT_embeddings_predictions_7.csv']

method_name = ['GCN method:','GraphSAGE method:','CSS-HGAT-SSP method:']

for i in range (len(file_list)):
    # Step 1: Load node features and label data.
    df = pd.read_csv("../experimental_results./(2)_embeddings/" + file_list[i], index_col=0)
    preds = df['predicted_label'].values
    embeddings = df.drop(columns=['predicted_label']).values
    node_ids = df.index.tolist()

    ## Load geographic unit IDs for training.
    nodes_train = pd.read_csv('../experimental_data/Grid500_AllCity_Nodes.csv')
    ## Filter out nodes with missing labels and convert labels to integers.
    node_labels = {row['ORIG_FID']: int(row['newLabel']) for _, row in nodes_train.iterrows() if pd.notnull(row['newLabel'])}

    # Step 2: Compute evaluation metrics.
    true_labels = []
    predicted_labels = []

    ## Load geographic unit IDs covered by the flood monitoring image.
    nodes_selected = pd.read_csv('./experimental_data/Grid500_Selected_FloodArea.csv')
    all_selected = nodes_selected["ORIG_FID"]

    ## Load geographic unit IDs used for training.
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
        predicted_labels.append(preds[node_id])

    ## Convert to numpy array.
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)

    #Step 3: Compute evaluation metrics.
    print(method_name[i])

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
