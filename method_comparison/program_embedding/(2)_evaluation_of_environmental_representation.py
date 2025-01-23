'''Evaluating environmental representations obtained by different methods.'''

import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
from sklearn import metrics
from sklearn.metrics import fowlkes_mallows_score


def entropy(labels):
    # Calculate the entropy of a set of labels.
    label_counts = np.bincount(labels)
    probabilities = label_counts / len(labels)
    return -np.sum(probabilities * np.log2(probabilities))

file_list =['DeepWalk_embeddings.csv','Node2Vec_embeddings.csv','GAE_embeddings.csv','GCN_embeddings.csv','GraphSAGE_embeddings.csv','GAT_embeddings_predictions_7.csv']

for file_path in file_list:
    embeddings_df = pd.read_csv("../experimental_results/(2)_embeddings/" + file_path, index_col=0)
    node_embeddings = embeddings_df.values

    kmeans = KMeans(n_clusters=3, random_state=42).fit(node_embeddings)
    cluster_labels = kmeans.labels_

    # Calculate the FMI (Fowlkes-Mallows Index).
    nodes_selected = pd.read_csv('../experimental_data/Grid500_Selected_FloodArea.csv')
    all_selected = nodes_selected["ORIG_FID"]

    nodes_train = pd.read_csv('../experimental_data/Grid500_AllCity_Nodes.csv')
    orgin_indicate = nodes_train['Flood_Sum']
    train_nodes = nodes_train.loc[nodes_train['newLabel'].notna(), 'ORIG_FID'].values

    node_to_cluster = {node_id: cluster_label for node_id, cluster_label in
                       zip(nodes_selected["ORIG_FID"], cluster_labels)}
    true_labels = []
    predicted_labels = []
    for node_id in all_selected:
        if orgin_indicate.loc[nodes_train['ORIG_FID'] == node_id].values[0] <= 50:
            true_labels.append(0)
        elif orgin_indicate.loc[nodes_train['ORIG_FID'] == node_id].values[0] <= 500:
            true_labels.append(1)
        else:
            true_labels.append(2)
        predicted_labels.append(node_to_cluster.get(node_id, -1))

    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)

    fmi = fowlkes_mallows_score(true_labels, predicted_labels)
    print(f'Fowlkes-Mallows Index: {fmi:.4f}')

    # Calculate the average entropy.
    cluster_entropies = [entropy(predicted_labels[true_labels == l]) for l in np.unique(true_labels)]
    average_entropy = np.mean(cluster_entropies)
    print(f'Average Entropy: {average_entropy:.4f}')





