"""https://github.com/diplomacy/research/blob/master/diplomacy_research/models/layers/graph_convolution.py#L28"""
import numpy as np


def preprocess_adjacency(adjacency_matrix):
    """ Symmetrically normalize the adjacency matrix for graph convolutions.
        :param adjacency_matrix: A NxN adjacency matrix
        :return: A normalized NxN adjacency matrix
    """
    # Computing A^~ = A + I_N
    adj = adjacency_matrix
    adj_tilde = adj + np.eye(adj.shape[0])

    # Calculating the sum of each row
    sum_of_row = np.array(adj_tilde.sum(1))

    # Calculating the D tilde matrix ^ (-1/2)
    d_inv_sqrt = np.power(sum_of_row, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)

    # Calculating the normalized adjacency matrix
    norm_adj = adj_tilde.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return np.array(norm_adj, dtype=np.float32)
