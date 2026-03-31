from torch_geometric.datasets import Planetoid

def load_data():
    dataset = Planetoid(root='data/Planetoid', name='Cora')
    return dataset[0]