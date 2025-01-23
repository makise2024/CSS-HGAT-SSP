'''Calculating the environmental representation of grid cells based on the DeepWalk method.'''

import pandas as pd
import networkx as nx
from gensim.models import Word2Vec
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, silhouette_score

def deepwalk(graph, num_walks, walk_length):
    # Perform multiple random walks starting from each node in the graph.
    walks = []
    nodes = list(graph.nodes())
    for _ in range(num_walks):
        np.random.shuffle(nodes)
        for node in nodes:
            walks.append(random_walk(graph, node, walk_length))
    return walks

def random_walk(graph, start_node, walk_length):
    # Generate a single random walk sequence starting from the given node.
    walk = [start_node]
    for _ in range(walk_length - 1):
        neighbors = list(graph.neighbors(walk[-1]))
        if len(neighbors) > 0:
            walk.append(np.random.choice(neighbors))
        else:
            break
    return walk

def get_node_embeddings(graph, num_walks, walk_length, dimensions):
    # Generate node embeddings using the DeepWalk algorithm and Word2Vec model.
    walks = deepwalk(graph, num_walks, walk_length)
    walks = [[str(node) for node in walk] for walk in walks]  # Convert to string for Word2Vec
    model = Word2Vec(walks, vector_size=dimensions, window=5, min_count=0, sg=1, workers=4)
    embeddings = {int(node): model.wv[str(node)] for node in graph.nodes()}
    return embeddings



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
num_walks = 4
walk_length = 8
dimensions = 128
node_embeddings = get_node_embeddings(G, num_walks, walk_length, dimensions)

# Output the final node representations.
embeddings_matrix = np.zeros((len(G.nodes()), dimensions))
node_ids = list(G.nodes())
for node_id in node_ids:
    embeddings_matrix[node_id] = node_embeddings[node_id]
embeddings_df = pd.DataFrame(embeddings_matrix, index=node_ids)
embeddings_df.to_csv('../experimental_results/(2)_embeddings/DeepWalk_embeddings.csv')