import torch
import yaml
import argparse
from src.data.dataset import load_data
from src.utils.graph_utils import compute_normalized_adjacency
from src.experiments import experiments
from src.utils.plotting import plot
from src.experiment_tsne import tsne

def main(dataset = 'Cora'):

    print(f"Dataset used: {dataset}")
    data = load_data(dataset)

    A_hat = compute_normalized_adjacency(data.edge_index, data.num_nodes)
    A_hat_torch = torch.sparse_coo_tensor(
        A_hat.nonzero(),
        A_hat.data,
        A_hat.shape,
        dtype=torch.float32
    )

    with open(f"config/{dataset}_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    epochs = config["training"]["epochs"]

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)


    x = data.x.to(device)
    y = data.y.to(device)
    A_hat_torch = A_hat_torch.to(device)

    num_nodes = data.num_nodes
    num_edges = data.num_edges
    num_features = data.num_features
    num_classes = data.y.unique().size(0)

    print(f"Dataset has {num_nodes} nodes.")
    print(f"Dataset has {num_edges} edges.")
    print(f"Each node has {num_features} features.")
    print(f"Dataset has {num_classes} classes.")

    
    train_mask = data.train_mask
    val_mask = data.val_mask
    test_mask = data.test_mask


    results = experiments(x,y, num_features, num_classes, train_mask, val_mask, test_mask, A_hat_torch, device, dataset)

    plot(results, epochs, dataset) 

    tsne(x,y, num_features, num_classes, train_mask, val_mask, test_mask, A_hat_torch, device, dataset)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Cora", help="Dataset name")
    args = parser.parse_args()

    dataset = args.dataset
    main(dataset)