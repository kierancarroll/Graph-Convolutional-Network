import yaml
import torch.nn as nn
import torch.optim as optim
from src.models.GCN import GCN
from src.training import train
from src.test import test


def experiments(x,y, num_features, num_classes, train_mask, val_mask, test_mask, A_hat_torch, device, dataset):

    with open(f"config/{dataset}_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    lr = config["training"]["lr"]
    epochs = config["training"]["epochs"]
    N_RUNS = config["training"]["n_runs"]
    hidden_dim = config["model"]["hidden_dim"]

    results = {
    "baseline": {"train_loss": [], "val_acc": [], "test_acc": []},
    "l2":  {"train_loss": [], "val_acc": [], "test_acc": []},
    "dropout":       {"train_loss": [], "val_acc": [], "test_acc": []},
    "full":     {"train_loss": [], "val_acc": [], "test_acc": []},
    }
    
    # experiment 1: baseline
    for run in range(N_RUNS):
        model = GCN(num_features, hidden_dim, num_classes, 0).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0)
        criterion = nn.CrossEntropyLoss()

        train_loss, val_acc =train(model, optimizer, criterion, epochs, x, y, train_mask, val_mask, A_hat_torch)

        test_acc = test(model, x, y, test_mask, A_hat_torch)

        print(f" (BASELINE) Training Complete! | Test Accuracy: {test_acc} | Final Validation Accuracy: {val_acc[-1]}")

        results["baseline"]["train_loss"].append(train_loss)
        results["baseline"]["val_acc"].append(val_acc)
        results["baseline"]["test_acc"].append(test_acc)

    # experiment 2: l2 reg
    for run in range(N_RUNS):
        model = GCN(num_features, hidden_dim, num_classes, 0).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()

        train_loss, val_acc = train(model, optimizer, criterion, epochs, x, y, train_mask, val_mask, A_hat_torch)

        test_acc = test(model, x, y, test_mask, A_hat_torch)

        print(f" (L2) Training Complete! | Test Accuracy: {test_acc} | Final Validation Accuracy: {val_acc[-1]}")

        results["l2"]["train_loss"].append(train_loss)
        results["l2"]["val_acc"].append(val_acc)
        results["l2"]["test_acc"].append(test_acc)

    # experiment 3: dropout
    for run in range(N_RUNS):
        model = GCN(num_features, hidden_dim, num_classes, 0.5).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0)
        criterion = nn.CrossEntropyLoss()

        train_loss, val_acc = train(model, optimizer, criterion, epochs, x, y, train_mask, val_mask, A_hat_torch)

        test_acc = test(model, x, y, test_mask, A_hat_torch)

        print(f" (DROPOUT) Training Complete! | Test Accuracy: {test_acc} | Final Validation Accuracy: {val_acc[-1]}")

        results["dropout"]["train_loss"].append(train_loss)
        results["dropout"]["val_acc"].append(val_acc)
        results["dropout"]["test_acc"].append(test_acc)

    # experiment 4: l2 reg & dropout
    for run in range(N_RUNS):
        model = GCN(num_features, hidden_dim, num_classes, 0.5).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()

        train_loss, val_acc = train(model, optimizer, criterion, epochs, x, y, train_mask, val_mask, A_hat_torch)

        test_acc = test(model, x, y, test_mask, A_hat_torch)

        print(f" (FULL) Training Complete! | Test Accuracy: {test_acc} | Final Validation Accuracy: {val_acc[-1]}")

        results["full"]["train_loss"].append(train_loss)
        results["full"]["val_acc"].append(val_acc)
        results["full"]["test_acc"].append(test_acc)

    return results