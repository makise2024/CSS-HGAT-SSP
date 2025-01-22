'''Calculating the distribution of silhouette coefficients for clustering results based on different representations.'''

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_samples
from sklearn.cluster import KMeans
import pandas as pd


# Importing necessary libraries for handling font issues
import matplotlib.font_manager as fm

# Getting a list of available fonts and finding one that supports Chinese characters
available_fonts = fm.findSystemFonts()
chinese_font = next((font for font in available_fonts if 'SimHei' in font), None)

file_list =['DeepWalk_embeddings.csv','Node2Vec_embeddings.csv','GAE_embeddings.csv','GCN_embeddings.csv','GraphSAGE_embeddings.csv','GAT_embeddings_predictions_7.csv']

fig_title = ['(a) DeepWalk','(b) Node2Vec','(c) GAE','(d) GCN','(e) GraphSAGE','(f) CSS-HGAT (our method)']

for i in range(len(file_list)):
    embeddings_df = pd.read_csv("../experimental_results/(2)_embeddings/" + file_list[i], index_col=0)
    node_embeddings = embeddings_df.values

    kmeans = KMeans(n_clusters=3, random_state=42).fit(node_embeddings)
    cluster_labels = kmeans.labels_

    # Calculate the Silhouette Coefficient.
    silhouette_vals = silhouette_samples(node_embeddings, cluster_labels)

    # Plot the Distribution of Silhouette Coefficients.
    plt.figure(figsize=(10, 7),dpi=300)
    plt.rcParams['axes.unicode_minus'] = False
    sns.histplot(silhouette_vals, kde=True, bins=30)
    plt.title(fig_title[i], fontproperties=fm.FontProperties(fname=chinese_font, size=28))

    plt.xticks(fontproperties=fm.FontProperties(fname=chinese_font, size=24))
    plt.yticks(fontproperties=fm.FontProperties(fname=chinese_font, size=24))

    plt.subplots_adjust(left=0.075, right=0.995, top=0.945, bottom=0.05)
    plt.show()


