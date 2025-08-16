import random

import math
import torch.nn as nn
import torch
from math import log
import pandas as pd
import numpy as np
EPS = 1e-8
from statistics import NormalDist
from sklearn.decomposition import PCA
from sklearn_extra.cluster import KMedoids

def cluster_features(features: pd.DataFrame, group_num=3):
    # Cluster the given features into {group_num} groups
    # The first column of features should be the feature name, the second column should be miu,
    # and the third column should be sigma
    n = len(features)
    distance_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            distance_matrix[i, j] = 1 - calculate_overlap(features.iloc[i, 1], features.iloc[i, 2], features.iloc[j, 1], features.iloc[j, 2])

    clustered_df = features.copy()

    kmedoids = KMedoids(n_clusters=group_num, metric='precomputed', random_state=0)
    # Fit the model
    kmedoids.fit(distance_matrix)
    # Get the clustering labels
    clustered_df['label'] = kmedoids.labels_

    min_indices = clustered_df.reset_index().groupby('label')['index'].min()

    # Then, sort the labels according to these minimum indices
    sorted_labels = min_indices.sort_values().index

    # Create a mapping to map old labels to new labels
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(sorted_labels)}
    clustered_df['label'] = clustered_df['label'].map(label_mapping)

    return clustered_df


def get_embsize_by_vocab(feature_map):
    size_map = dict()
    for feature_name, feature_info in feature_map.features.items():
        if feature_info.get('vocab_size', None) is not None:
            size_map[feature_name] = int(feature_info['vocab_size']**0.25)
    return size_map

def get_embsize_by_pca(cur_model,ratio=0.95):
    size_map = dict()
    emb_dict = cur_model.embedding_layer.embedding_layer.embedding_layers
    for feature_name, emb in emb_dict.items():
        cur_emb = emb.weight.detach().cpu().numpy()
        pca = PCA(n_components=ratio)
        pca.fit(cur_emb)
        size_map[feature_name] = pca.n_components_
    return size_map

def calculate_overlap(mu1, sigma1, mu2, sigma2):
    return NormalDist(mu=mu1, sigma=sigma1).overlap(NormalDist(mu=mu2, sigma=sigma2))


def get_gates_prob_dfo(gates_theta, gates_sigma, tau = None):
    assert len(gates_theta) == len(gates_sigma), "gates_theta and gates_sigma should have the same length"
    gates_prob = gates_theta.clone()
    for i in range(gates_theta.shape[0]):
        gates_prob[i] = get_prob_dfo(gates_theta[i], gates_sigma[i], tau = tau)
    return gates_prob

def get_prob_dfo(unit, sigma_unit, tau = None):
    if tau is None:
        tau = 0.5

    # Reparameterization Trick
    eps = torch.randn(1)*sigma_unit

    return torch.sigmoid((1.0 / tau) * (unit + eps))

def get_sum_feature_dimensions(embedding_layer):
    total_dimensions = 0
    for layer in embedding_layer.embedding_layer.embedding_layers.values():
        if isinstance(layer,nn.Embedding):
            total_dimensions += layer.embedding_dim
        elif isinstance(layer,nn.Linear):
            total_dimensions += layer.out_features
        else:
            raise TypeError(f"Unsupported layer type {type(layer)} for layer {name}")
    return total_dimensions

def create_unique_vector(a, b):
    if b > a:
        raise ValueError("b cannot be greater than a.")
    random_numbers = random.sample(range(a), b)
    vector = np.array(random_numbers)
    return vector.tolist()

def permute_feature(data_generator, feature_idx):
    """
    Permutes the values of a specific feature in each batch produced by the data_generator.

    Args:
    - data_generator: Original data generator.
    - feature_idx: The index of the feature you want to permute.

    Yields:
    - Batch with permuted feature values.
    """
    if not isinstance(feature_idx, list):
        feature_idx = [feature_idx]

    for batch in data_generator:
        # Deep copy to avoid modifying the original batch
        permuted_batch = batch.clone()

        # Permute the feature using PyTorch functions
        perm = torch.randperm(permuted_batch.size(0))
        permuted_batch[:, feature_idx] = permuted_batch[perm, :][:, feature_idx]

        yield permuted_batch