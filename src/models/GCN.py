import torch.nn as nn
import torch.nn.functional as F
from .GCN_layer import GCNLayer

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super().__init__()
        
        self.gcn1 = GCNLayer(input_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, output_dim)
        self.dropout = dropout

    def forward(self, X, A_hat):
        X = self.gcn1(X, A_hat)
        X = F.relu(X)
        X = F.dropout(X, p=self.dropout, training=self.training)
        X = self.gcn2(X, A_hat)
        return X