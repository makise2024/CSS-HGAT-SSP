'''Comparing the clustering performance based on different environmental representations.'''

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import silhouette_score,jaccard_score
from sklearn.metrics import adjusted_mutual_info_score
import numpy as np
from sklearn.metrics.pairwise import cosine_distances

def calculate_alignment_loss(embeddings, labels):
    """
    Compute Alignment Loss
    :param embeddings: Node embeddings with shape (N, d)
    :param labels: Node labels with shape (N,)
    :return: Alignment Loss
    """
    same_cosine_dists = []
    diff_cosine_dists = []
    same_class_dists = []
    diff_class_dists = []

    unique_labels = np.unique(labels)
    for label in unique_labels:
        same_class_indices = np.where(labels == label)[0]
        diff_class_indices = np.where(labels != label)[0]

        same_class_emb = embeddings[same_class_indices]
        diff_class_emb = embeddings[diff_class_indices]

        for i in range(len(same_class_emb)):
            for j in range(i + 1, len(same_class_emb)):
                dist = cosine_distances(same_class_emb[i].reshape(1, -1), same_class_emb[j].reshape(1, -1))[0][0]
                same_cosine_dists.append(dist)

        for i in range(len(same_class_emb)):
            for j in range(len(diff_class_emb)):
                dist = cosine_distances(same_class_emb[i].reshape(1, -1), diff_class_emb[j].reshape(1, -1))[0][0]
                diff_cosine_dists.append(dist)

        for i in range(len(same_class_emb)):
            for j in range(i + 1, len(same_class_emb)):
                dist = np.linalg.norm(same_class_emb[i] - same_class_emb[j])
                same_class_dists.append(dist)

        for i in range(len(same_class_emb)):
            for j in range(len(diff_class_emb)):
                dist = np.linalg.norm(same_class_emb[i] - diff_class_emb[j])
                diff_class_dists.append(dist)

    avg_same_class_dist = np.mean(same_class_dists)
    avg_diff_class_dist = np.mean(diff_class_dists)

    alignment_loss = avg_same_class_dist / avg_diff_class_dist
    return same_cosine_dists,diff_cosine_dists,alignment_loss


def local_uniformity(embeddings, k=5):
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)
    local_std_distances = np.std(distances[:, 1:], axis=1)
    avg_local_std_distance = np.mean(local_std_distances)
    return avg_local_std_distance


file_list =['DeepWalk_embeddings.csv','Node2Vec_embeddings.csv','GAE_embeddings.csv','GCN_embeddings.csv','GraphSAGE_embeddings.csv','GAT_embeddings_predictions_7.csv']

nodes_df = pd.read_csv('../experimental_data/Grid500_AllCity_Nodes.csv')

# Compute the evaluation metrics for the representations obtained by each method.
for file_path in file_list:
    embeddings_df = pd.read_csv("../experimental_results/(2)_embeddings/" + file_path, index_col=0)
    node_embeddings = embeddings_df.values

    local_uniform = local_uniformity(node_embeddings, k=5)
    print(len(node_embeddings))
    print(f"Local Uniformity: {local_uniform}")

    labels = nodes_df['newLabel'].apply(lambda x: int(x) if pd.notna(x) and x in [0, 1, 2] else -1).values
    labeled_mask = labels != -1
    labeled_embeddings = node_embeddings[labeled_mask]
    true_labels = labels[labeled_mask]

    same_cosine_dists, diff_cosine_dists, alignment_loss = calculate_alignment_loss(labeled_embeddings, true_labels)
    print(f'Alignment Loss: {alignment_loss:.4f}')

