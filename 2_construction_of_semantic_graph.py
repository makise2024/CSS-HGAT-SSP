'''Clustering evaluation with semantic graphs of varying nearest neighbors: the proposed CSS-HGAT-SSP method'''

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import kneighbors_graph
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (precision_score, recall_score, roc_auc_score)
from sklearn.preprocessing import label_binarize

# Step 1: Read node representations and label data.
df = pd.read_csv('./experimental_results/(1)_neighborhood_selection/GAT_embeddings_predictions_7.csv', index_col=0)
embeddings = df.drop(columns=['predicted_label']).values
node_ids = df.index.tolist()

## Read the geographic unit numbers used for training
nodes_train = pd.read_csv('./experimental_data/Grid500_AllCity_Nodes.csv')
## Filter out entries where the newLabel field is empty and ensure its values are of type int.
node_labels = {row['ORIG_FID']: int(row['newLabel']) for _, row in nodes_train.iterrows() if pd.notnull(row['newLabel'])}


for k in range(1,11):
    # Step 2: Create the KNN graph
    adj_matrix = kneighbors_graph(embeddings, k, mode='connectivity', include_self=True)
    G = nx.from_scipy_sparse_array(adj_matrix)
    # print(f"Number of nodes in graph G: {G.number_of_nodes()}")

    # Step 3: Extract neighborhood features of nodes.
    degree = np.array([val for (node, val) in G.degree()])
    clustering_coefficient = np.array(list(nx.clustering(G).values()))

    ## Normalize neighborhood features.
    scaler = StandardScaler()
    degree_scaled = scaler.fit_transform(degree.reshape(-1, 1)).flatten()
    clustering_coefficient_scaled = scaler.fit_transform(clustering_coefficient.reshape(-1, 1)).flatten()

    # Combine node representations and neighborhood features.
    combined_features = np.hstack(
        (embeddings, degree_scaled.reshape(-1, 1), clustering_coefficient_scaled.reshape(-1, 1)))

    # Step 4: Clustering.
    num_clusters = len(set(node_labels.values()))  # 假设聚类数目等于标签种类数
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(combined_features)
    cluster_labels = kmeans.labels_

    # Step 5: Label propagation.
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

    # Save the cluster result.
    cluster_data = pd.DataFrame({
        'node_id': node_ids,
        'cluster_label': cluster_labels,
        'final_label': final_node_labels
    })
    filepath = "./experimental_results/(3)_knn_cluster/"  + f"clustering_results_k{k}.csv"
    cluster_data.to_csv(filepath, index=False, encoding='utf-8') #The best results are used to plot Fig. 13.

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
        predicted_labels.append(final_node_labels.get(node_id, -1))  # 使用get方法避免KeyError

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


