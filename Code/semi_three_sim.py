# -*- coding: utf-8 -*-
"""
KNN graph + Measure Propagation (semi-supervised) with CCER metric.
- Uses example-relative paths.
- English comments and logs.
- Saves per-iteration results and metrics (including CCER).

Run prerequisites:
  - XLSX files exported by your MST script:
      outputs/example/Groundnut_mst_{Euclidean,Manhattan,Cosine}.xlsx
  - CSV file with ground-truth labels:
      example/Groundnut_Oil_semi.csv
  - Output directory will be created:
      outputs/example/knn_mp_results/
"""

import os
from pathlib import Path
from queue import PriorityQueue

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import networkx as nx
from adjustText import adjust_text
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
)
from sklearn.metrics.pairwise import cosine_distances
from sklearn.model_selection import ParameterGrid
from sklearn.base import BaseEstimator, ClassifierMixin
import logging

# -----------------------------
# Graph and KNNGraph utilities
# -----------------------------
class Graph(dict):
    NEIGHBOURS_KEY = 'neighbours'
    WEIGHTS_KEY = 'weights'

    @property
    def vertices(self):
        return self.keys()

    @property
    def nb_vertices(self):
        return len(self)

    def add_edge(self, vertex, neighbour, weight=1):
        if vertex not in self:
            self[vertex] = {self.NEIGHBOURS_KEY: [neighbour],
                            self.WEIGHTS_KEY: [weight]}
        elif neighbour not in self[vertex][self.NEIGHBOURS_KEY]:
            self[vertex][self.NEIGHBOURS_KEY].append(neighbour)
            self[vertex][self.WEIGHTS_KEY].append(weight)

    def build(self, *args):
        self._build(*args)
        self.post_build_hook()
        return self

    def _build(self, edges):
        for source_id, dest_id in edges:
            if source_id != dest_id:
                self.add_edge(source_id, dest_id)
                self.add_edge(dest_id, source_id)

    def post_build_hook(self):
        for _, edge_info in self.items():
            edge_info[self.WEIGHTS_KEY] = np.expand_dims(np.array(edge_info[self.WEIGHTS_KEY]), 1)

# Similarities in (0,1], built from distances
def euc_similarity(x, y):
    return 1 / (1 + np.linalg.norm(x - y))

def manhattan_similarity(x, y):
    return 1 / (1 + np.sum(np.abs(x - y)))

def cosine_similarity_bounded(x, y):
    # cosine_distances returns 1 - cosine_similarity
    return 1 / (1 + cosine_distances([x], [y])[0][0])

class KNNGraph(Graph):
    def __init__(self, K, similarity_func, *args, **kwargs):
        super(KNNGraph, self).__init__(*args, **kwargs)
        self.K = K
        self.similarity_func = similarity_func

    class Neighbour:
        def __init__(self, id_, similarity):
            self.id = id_
            self.similarity = similarity

        def __lt__(self, other):
            return self.similarity < other.similarity

    def _build(self, X):
        """
        Build a directed KNN graph using the provided similarity function.
        """
        n_samples = len(X)
        for i in range(n_samples):
            neighbours = PriorityQueue(maxsize=self.K)
            for j in range(n_samples):
                if i != j:
                    sim = self.similarity_func(X[i], X[j])
                    neighbour = KNNGraph.Neighbour(j, sim)
                    if not neighbours.full():
                        neighbours.put(neighbour)
                    else:
                        lowest_entry = neighbours.get()
                        if neighbour.similarity > lowest_entry.similarity:
                            neighbours.put(neighbour)
                        else:
                            neighbours.put(lowest_entry)
            while not neighbours.empty():
                neighbour = neighbours.get()
                self.add_edge(i, neighbour.id, weight=neighbour.similarity)

# -----------------------------
# Measure Propagation (MP)
# -----------------------------
logger = logging.getLogger(__name__)

class MeasurePropagation:
    """
    Implements the Measure Propagation algorithm.
    """
    def __init__(self, mu=0.1, nu=0.01, tol=2e-2, max_iter=100):
        self.graph = None
        self.r, self.nb_classes = None, None
        self.p, self.q = None, None
        self.mu = mu
        self.nu = nu
        self.tol = tol
        self.max_iter = max_iter
        self.SMALL = 1e-10  # to avoid log(0)

    def _labels_to_probabilities(self, vertices_labels_dct):
        """
        Convert label dict (vertex -> label) into one-hot probabilities.
        Labels are remapped to consecutive indices [0, C-1].
        """
        unique_labels = np.unique(list(vertices_labels_dct.values()))
        label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
        nb_classes = len(unique_labels)

        probs = {}
        for vertex, label in vertices_labels_dct.items():
            probs[vertex] = np.zeros(nb_classes)
            probs[vertex][label_to_index[label]] = 1
        return probs, nb_classes

    def _init_probability_distributions(self):
        p = np.full((self.graph.nb_vertices, self.nb_classes), 1 / self.nb_classes)
        q = np.full((self.graph.nb_vertices, self.nb_classes), 1 / self.nb_classes)
        return p, q

    def compute_p_update(self, vertex):
        neighbours = self.graph[vertex][self.graph.NEIGHBOURS_KEY]
        w_neighbours = self.graph[vertex][self.graph.WEIGHTS_KEY]
        gamma = self.nu + self.mu * w_neighbours.sum()
        p_new = np.exp((np.log(self.q[neighbours] + self.SMALL) * w_neighbours).sum(axis=0) * (self.mu / gamma))
        return p_new / p_new.sum()

    def compute_p_updates(self):
        for vertex in self.graph.vertices:
            self.p[vertex] = self.compute_p_update(vertex)

    def compute_q_update(self, vertex):
        neighbours = self.graph[vertex][self.graph.NEIGHBOURS_KEY]
        w_neighbours = self.graph[vertex][self.graph.WEIGHTS_KEY]
        divident_right_sum = self.mu * (w_neighbours * self.p[neighbours]).sum(axis=0)
        divident_left_sum  = self.r[vertex] if vertex in self.r else 0
        divisor_right_sum  = self.mu * w_neighbours.sum()
        divisor_left_sum   = 1 if vertex in self.r else 0
        return (divident_right_sum + divident_left_sum) / (divisor_right_sum + divisor_left_sum)

    def compute_q_updates(self):
        q_new = np.zeros(self.q.shape)
        for vertex in self.graph.vertices:
            q_new[vertex] = self.compute_q_update(vertex)
        return q_new

    def alternate_minimization_step(self):
        self.compute_p_updates()
        q_new = self.compute_q_updates()
        test_convergence = self.compute_test_convergence(q_new)
        logger.debug("Test convergence %f", test_convergence)
        self.q = q_new
        return test_convergence

    def compute_test_convergence(self, q_new):
        div = q_new / (self.q + self.SMALL)
        beta = np.log(np.max(div, 1) + self.SMALL)
        accum = .0
        for vertex in self.graph.vertices:
            delta = 1 if vertex in self.r else 0
            d_i = np.array(self.graph[vertex][self.graph.WEIGHTS_KEY]).sum()
            accum += (delta + d_i) * beta[vertex]
        return accum

    def optimize(self, graph, vertices_labels_dct):
        """
        Run MP on a prebuilt graph and a dict of labeled vertices.
        """
        self.graph = graph
        self.r, self.nb_classes = self._labels_to_probabilities(vertices_labels_dct)
        self.p, self.q = self._init_probability_distributions()

        convergences = []
        for it in range(self.max_iter):
            logger.debug("Iteration %d", it)
            convergences.append(self.alternate_minimization_step())
            if it > 0:
                change = (convergences[it - 1] - convergences[it]) / max(convergences[it], self.SMALL)
                logger.debug("Change in convergence criteria %f", change)
                if change <= self.tol:
                    break

        logger.info("Convergence criterion reached on iteration %d", it)
        logger.info("Relative objective value: %f", convergences[it])

    def get_output_labels(self):
        return np.argmax(self.q, axis=1)

class MeasurePropagationSklearn(MeasurePropagation, BaseEstimator, ClassifierMixin):
    def fit(self, X_graph, y):
        """
        X_graph: a built Graph (e.g., KNNGraph().build(X))
        y: label vector with -1 for unlabeled samples
        """
        labeled_indices = np.where(y != -1)[0]
        vertices_labels_dct = {idx: y[idx] for idx in labeled_indices}
        return self.optimize(X_graph, vertices_labels_dct)

    def predict(self):
        return self.get_output_labels()

# -----------------------------
# Helper functions
# -----------------------------
def plot_and_save_graph(results_df, K, params, iteration, save_dir, file_path):
    """
    Draw the KNN graph with node texts: ID + original/pred labels, then save PNG.
    """
    G = nx.Graph()

    # Map vertex -> name for display (from 'name' column in results_df)
    vertex_to_name = {row['Vertex']: row['name'] for _, row in results_df.iterrows()}

    # Nodes with attributes
    for vertex in results_df['Vertex']:
        G.add_node(
            vertex,
            label=results_df[results_df['Vertex'] == vertex]['OriginalLabel'].iloc[0],
            pred_label=results_df[results_df['Vertex'] == vertex]['PredictedLabel'].iloc[0]
        )

    # Edges from neighbor lists
    for _, row in results_df.iterrows():
        vertex = row['Vertex']
        for neighbor in row['Neighbors']:
            G.add_edge(vertex, neighbor)

    plt.figure(figsize=(20, 16), dpi=400)
    pos = nx.spring_layout(G, seed=0)
    pred_labels = nx.get_node_attributes(G, 'pred_label')
    color_map = ['#1f78b4' if pred_labels[node] == 0 else '#33a02c' for node in G]
    nx.draw(G, pos, node_color=color_map, with_labels=False, node_size=500,
            edge_color='k', linewidths=1, font_size=12)

    # Annotate each node with ID + (O,P)
    texts = [
        plt.text(x, y, f'ID:{vertex_to_name[vertex]}\nO:{G.nodes[vertex]["label"]}, P:{G.nodes[vertex]["pred_label"]}',
                 fontsize=12, ha='right')
        for vertex, (x, y) in pos.items()
    ]
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='grey'))

    blue_patch = plt.Line2D([0], [0], marker='o', color='w',
                            markerfacecolor='#1f78b4', markersize=15, label='Class 0')
    green_patch = plt.Line2D([0], [0], marker='o', color='w',
                             markerfacecolor='#33a02c', markersize=15, label='Class 1')
    plt.legend(handles=[blue_patch, green_patch], loc='lower right', fontsize='large')

    plt.title("KNN Graph Visualization with Original and Predicted Labels", y=0.95,
              fontdict={'fontsize': 20, 'fontweight': 'normal', 'color': 'black'})

    base_name = Path(file_path).stem
    filename = f"{base_name}_K{K}_iteration{iteration}.png"
    save_path = Path(save_dir) / filename
    plt.savefig(save_path)
    plt.close()

# -----------------------------
# Core processing
# -----------------------------
def process_file(file_path_xlsx, file_path_labels_csv, K_values, param_grid, save_dir):
    """
    file_path_xlsx: MST-marked XLSX produced earlier (per metric).
    file_path_labels_csv: CSV containing true labels for evaluation.
    """
    data = pd.read_excel(file_path_xlsx)
    data1 = pd.read_csv(file_path_labels_csv)

    # Feature matrix (skip first column like 'Wavelength', and last column 'Class')
    X = data.iloc[:, 1:-1]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Labels
    Y = data.iloc[:, -1]     # 'Class' with -1 for unlabeled
    X = X.to_numpy()
    Y = Y.to_numpy()

    # Choose similarity function based on filename
    fp_str = str(file_path_xlsx)
    if 'Euclidean' in fp_str:
        similarity_func = euc_similarity
    elif 'Manhattan' in fp_str:
        similarity_func = manhattan_similarity
    elif 'Cosine' in fp_str:
        similarity_func = cosine_similarity_bounded
    else:
        raise ValueError("Unknown similarity function for the given file path name.")

    data_class = data['Class']              # may contain -1 for unlabeled
    data1_class = data1['Class']            # ground-truth labels
    data_labels = data['Wavelength']        # node display name

    iteration = 0
    for K in tqdm(K_values, desc="K Values Loop"):
        k_metrics_df = pd.DataFrame(columns=['iteration', 'accuracy', 'recall', 'precision', 'f1', 'auc', 'ccer'])

        for params in tqdm(ParameterGrid(param_grid), desc="Parameter Grid Loop"):
            print(f"\nProcessing with all chosen labels, K={K}, iteration: {iteration}")

            labeled_indices = np.where(data_class != -1)[0]
            unlabeled_indices = np.where(data_class == -1)[0]

            # Build y with -1 for unlabeled
            y_modified = data_class.copy()
            y_modified[unlabeled_indices] = -1

            # Build KNN graph on standardized X
            knn_graph = KNNGraph(K=K, similarity_func=similarity_func).build(X_scaled)

            # Run Measure Propagation
            mp = MeasurePropagationSklearn(**params)
            mp.fit(knn_graph, y_modified)
            output_labels = mp.predict()

            # Per-iteration metrics (prediction vs ground-truth)
            accuracy  = accuracy_score(data1_class, output_labels)
            recall    = recall_score(data1_class, output_labels, average='macro')
            precision = precision_score(data1_class, output_labels, average='macro')
            f1        = f1_score(data1_class, output_labels, average='macro')
            auc       = roc_auc_score(data1_class, output_labels, average='macro', multi_class='ovo')

            # Build results_df for saving and CCER computation
            neighbors_list = [knn_graph[vertex][knn_graph.NEIGHBOURS_KEY] for vertex in knn_graph.vertices]
            results_df = pd.DataFrame({
                'name': data_labels,
                'Vertex': list(knn_graph.vertices),
                'Neighbors': neighbors_list,
                'OriginalLabel': data_class,
                'PredictedLabel': output_labels
            })

            # --- CCER (Cross-Class Edge Ratio) ---
            # CCER = (# edges that link nodes with different predicted labels) / (total # edges)
            cross_edges = 0
            total_edges = 0
            # Note: if the graph is directed KNN, each (i->j) is counted once.
            # If you want undirected counting, you could half the denominator for symmetric edges.
            pred_label_map = dict(zip(results_df['Vertex'], results_df['PredictedLabel']))
            for _, row in results_df.iterrows():
                v_i = row['Vertex']
                y_i = row['PredictedLabel']
                for v_j in row['Neighbors']:
                    y_j = pred_label_map[v_j]
                    total_edges += 1
                    if y_i != y_j:
                        cross_edges += 1
            ccer = cross_edges / total_edges if total_edges > 0 else 0.0
            print(f"Accuracy/Recall/Precision/F1/AUC/CCER (K={K}, it={iteration}): "
                  f"{accuracy:.4f}, {recall:.4f}, {precision:.4f}, {f1:.4f}, {auc:.4f}, {ccer:.4f}")

            # Save metrics including CCER
            new_metrics = pd.DataFrame([{
                'iteration': iteration,
                'accuracy': accuracy,
                'recall': recall,
                'precision': precision,
                'f1': f1,
                'auc': auc,
                'ccer': ccer
            }])
            k_metrics_df = pd.concat([k_metrics_df, new_metrics], ignore_index=True)

            # Save per-iteration results with CCER column
            results_df['CCER'] = ccer
            csv_file_path = Path(save_dir) / f"results_{Path(file_path_xlsx).stem}_K{K}_iteration{iteration}.csv"
            results_df.to_csv(csv_file_path, index=False)

            # Plot and save graph image
            plot_and_save_graph(results_df, K, params, iteration, save_dir, file_path_xlsx)

            iteration += 1

        # Save per-K metrics (now includes CCER)
        metrics_csv_path = Path(save_dir) / f"metrics_{Path(file_path_xlsx).stem}_K{K}.csv"
        k_metrics_df.to_csv(metrics_csv_path, index=False)

# -----------------------------
# Paths & run
# -----------------------------
file_paths = [
    Path('outputs/example_mst_Euclidean.xlsx'),
    Path('outputs/example_mst_Manhattan.xlsx'),
    Path('outputs/example_mst_Cosine.xlsx'),
]
file_path1 = Path('example_semi.csv')

save_path = Path('outputs/example/knn_mp_results')
save_path.mkdir(parents=True, exist_ok=True)

K_values = [3, 4, 5]
param_grid = {
    'mu': [1e-8, 1e-4, 0.01, 0.1, 1, 10, 100],
    'nu': [1e-8, 1e-6, 1e-4, 0.01, 0.1],
    'max_iter': [1000],
    'tol': [1e-1]
}

for fp in file_paths:
    process_file(fp, file_path1, K_values, param_grid, save_path)
