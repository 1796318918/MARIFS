import numpy as np
from sklearn import neighbors
import scipy.sparse as sp
import torch

def feature_normalized(dt):
    mu = np.mean(dt, axis=1)
    sigma = np.std(dt, axis=1)
    return (dt - mu[:, np.newaxis]) / sigma[:, np.newaxis]


def create_graph_from_embedding(embedding, name, n):
    latent_dim, batch_size = embedding.shape
    if name == 'knn':
        A = neighbors.kneighbors_graph(embedding, n_neighbors = n).toarray()
        A = (A + np.transpose(A)) / 2
        return A


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool_)

def sim_matrix_euclidean_distance(X, sigma, threshold=0.5):  # 1.5
    # 计算欧氏距离矩阵
    dist_matrix = torch.cdist(X, X, p=2)
    # 使用高斯核
    sim_matrix = torch.exp(-dist_matrix ** 2 / (2 * sigma ** 2))
    # 将小于阈值的数置为0
    sim_matrix[sim_matrix <= threshold] = 0

    return sim_matrix

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            # mx = mx.tocoo()
            mx = sp.coo_matrix(mx)
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    mX0 = np.mean(features, axis=0)
    X1 = features - np.outer(np.ones(features.shape[0]), mX0)
    scal = 1. / np.sqrt(np.sum(X1 * X1, axis=0) + np.finfo(float).eps)
    features = X1 * scal

    # rowsum = np.array(features.sum(1))
    # # r_inv = np.power(rowsum, -1).flatten()
    # r_inv = (1. / rowsum).flatten()
    # r_inv[np.isinf(r_inv)] = 0.
    # r_mat_inv = sp.diags(r_inv)
    # features = r_mat_inv.dot(features)
    # # return sparse_to_tuple(features)
    return features

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    edge = np.array(np.nonzero(adj_normalized.todense()))
    # return sparse_to_tuple(adj_normalized), edge
    return adj_normalized, edge

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)