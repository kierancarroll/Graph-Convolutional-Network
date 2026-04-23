from torch_geometric.datasets import Planetoid

def load_data(dataset):
    if dataset == 'Cora':
        dataset = Planetoid(root='data/Planetoid', name='Cora')

    elif dataset == 'Citeseer':
        dataset = Planetoid(root='data/Planetoid', name='Citeseer')

    else: 
        assert dataset=="PubMed", "ERROR INVALID DATASET"
        dataset = Planetoid(root='data/Planetoid', name='PubMed')

    return dataset[0]