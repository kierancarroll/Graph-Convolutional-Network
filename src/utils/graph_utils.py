import numpy as np
import torch
from scipy.sparse import coo_matrix, diags

def compute_normalized_adjacency(edge_index, num_nodes):
    row, col = edge_index.numpy()
    
    adj = coo_matrix((np.ones(len(row)), (row, col)), shape=(num_nodes, num_nodes))
    
    adj = adj + diags(np.ones(num_nodes))
    
    degree = np.array(adj.sum(1)).flatten()
    d_inv_sqrt = np.power(degree, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    
    D_inv_sqrt = diags(d_inv_sqrt)
    
    A_hat = D_inv_sqrt @ adj @ D_inv_sqrt
    
    return A_hat