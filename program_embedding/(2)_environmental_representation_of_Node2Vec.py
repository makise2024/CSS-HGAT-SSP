'''Calculating the environmental representation of grid cells based on the Node2Vec method.'''

import pandas as pd
import networkx as nx
import numpy as np
from node2vec import Node2Vec
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, silhouette_score

# Read data.
nodes_df = pd.read_csv('../experimental_data/Grid500_AllCity_Nodes.csv')
edges_df = pd.read_csv('../experimental_data/Grid500_AllCity_Edges.csv')

# Create a graph.
G = nx.Graph()
for _, row in nodes_df.iterrows():
    node_id = int(row['ORIG_FID'])
    G.add_node(node_id)
for _, row in edges_df.iterrows():
    if int(row['TARGET_FID']) - int(row['JOIN_FID']):
        G.add_edge(int(row['TARGET_FID']), int(row['JOIN_FID']))

# Training process.
node2vec = Node2Vec(G, dimensions=128, walk_length=8, num_walks=4, workers=4)
model = node2vec.fit(window=10, min_count=1, batch_words=4)

# Output the final node representations.
node_embeddings = np.array([model.wv[str(node)] for node in G.nodes()])
node_ids = list(G.nodes())
embeddings_df = pd.DataFrame(node_embeddings, index=node_ids)
embeddings_df.to_csv('../experimental_results/(2)_embeddings/Node2Vec_embeddings.csv')


