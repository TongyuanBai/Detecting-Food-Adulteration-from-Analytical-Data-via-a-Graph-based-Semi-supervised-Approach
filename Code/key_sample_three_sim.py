# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances, cosine_distances
from scipy.sparse.csgraph import minimum_spanning_tree
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# Load CSV (use an example path; replace with your actual relative path if needed)
file_path = Path('example.csv')
df = pd.read_csv(file_path)

print(df.head())

# Prepare features and labels(Check the name of clumns)
data = df.drop(columns=['Wavelength', 'Class'])
labels = df['Class']

# Standardize features
scaler = StandardScaler()
data_standardized = scaler.fit_transform(data)

# Compute distance matrices
euclidean_dist_matrix = euclidean_distances(data_standardized)
manhattan_dist_matrix = manhattan_distances(data_standardized)
cosine_dist_matrix = cosine_distances(data_standardized)

# Ensure output directory exists (example path)
output_dir = Path('outputs/example')
output_dir.mkdir(parents=True, exist_ok=True)

def process_distance_matrix(dist_matrix, dist_name):
    """
    Process a given distance matrix:
      1) Print the matrix (as DataFrame)
      2) Build MST and visualize it
      3) Convert MST to adjacency matrix
      4) Spectral clustering on adjacency (n_clusters = 2)
      5) Select key nodes by degree >= 95th percentile (per original code)
      6) Mark selected nodes' labels; others set to -1
      7) Save the marked DataFrame to an Excel file under example outputs
    """
    # Convert to pandas DataFrame for quick inspection
    dist_df = pd.DataFrame(dist_matrix)
    print(f"\n{dist_name} Distance Matrix:")
    print(dist_df)

    # Compute Minimum Spanning Tree (MST)
    mst = minimum_spanning_tree(dist_matrix)

    # Build a NetworkX graph from MST and draw it
    G = nx.from_scipy_sparse_array(mst)
    pos = nx.spring_layout(G)
    plt.figure(figsize=(12, 12))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)
    plt.title(f'Minimum Spanning Tree ({dist_name})')
    plt.show()

    # Convert MST to dense array for inspection
    mst_dense = mst.toarray()

    # Print MST matrix
    mst_df = pd.DataFrame(mst_dense)
    print(f"\n{dist_name} MST Matrix:")
    print(mst_df)

    # Build adjacency matrix (1 if edge exists, else 0)
    adjacency_matrix = np.where(mst_dense > 0, 1, 0)

    # Spectral Clustering on the precomputed (binary) adjacency
    n_clusters = 2  # keep the original setting
    spectral = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', assign_labels='kmeans')
    labels_pred = spectral.fit_predict(adjacency_matrix)

    # Collect community nodes
    communities = {}
    for i in range(n_clusters):
        community_nodes = np.where(labels_pred == i)[0]
        communities[i] = community_nodes
        print(f"Community {i} ({dist_name}): {community_nodes}")

    # Build graph from adjacency for degree computation
    G = nx.Graph()
    for i in range(adjacency_matrix.shape[0]):
        for j in range(i + 1, adjacency_matrix.shape[1]):
            if adjacency_matrix[i, j] > 0:
                G.add_edge(i, j, weight=adjacency_matrix[i, j])

    # Compute degree and apply 95th percentile threshold
    all_degrees = np.array([degree for _, degree in G.degree()])
    degree_threshold = np.percentile(all_degrees, 95) if all_degrees.size > 0 else 0
    print(f"95th Percentile Degree Threshold ({dist_name}): {degree_threshold}")

    # Select nodes from each community whose degree >= threshold
    selected_nodes = {}
    for community, nodes in communities.items():
        if nodes.size == 0:
            selected_nodes[community] = []
            continue
        degrees = dict(G.degree(nodes))
        sorted_nodes = sorted(degrees, key=degrees.get, reverse=True)
        filtered_nodes = [node for node in sorted_nodes if degrees[node] >= degree_threshold]
        selected_nodes[community] = filtered_nodes

    # Print selected nodes per community
    for community, nodes in selected_nodes.items():
        print(f"Selected nodes from Community {community} ({dist_name}): {nodes}")

    # Create a new DataFrame to save marked data
    marked_df = df.copy()
    marked_df['Class'] = -1  # initialize as -1

    for nodes in selected_nodes.values():
        marked_df.loc[nodes, 'Class'] = df.loc[nodes, 'Class']

    # Save to Excel (example output path)
    output_file_path = output_dir / f"Groundnut_mst_{dist_name}.xlsx"
    marked_df.to_excel(output_file_path, index=False)
    print(f"{dist_name} labeled data saved to {output_file_path}")

# Process each distance type
process_distance_matrix(euclidean_dist_matrix, "Euclidean")
process_distance_matrix(manhattan_dist_matrix, "Manhattan")
process_distance_matrix(cosine_dist_matrix, "Cosine")
