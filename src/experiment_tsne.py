import yaml
import torch.nn as nn
import torch.optim as optim
from src.models.GCN import GCN
from src.training import train
from src.utils.embeddings_visuals import visualize_embeddings


def tsne(x,y, num_features, num_classes, train_mask, val_mask, test_mask, A_hat_torch, device, dataset):
    with open(f"config/{dataset}_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    lr = config["training"]["lr"]
    epochs = config["training"]["epochs"]
    hidden_dim = config["model"]["hidden_dim"]

    model = GCN(num_features, hidden_dim, num_classes, 0.5).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    train_loss, val_acc = train(model, optimizer, criterion, epochs, x, y, train_mask, val_mask, A_hat_torch)

    test_acc = visualize_embeddings(model, x, y, test_mask, A_hat_torch, dataset)

    print(f" (FULL) Training Complete! | Test Accuracy: {test_acc} | Final Validation Accuracy: {val_acc[-1]}")


    return