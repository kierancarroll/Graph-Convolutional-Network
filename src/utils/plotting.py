import torch
import numpy as np
import matplotlib.pyplot as plt


def plot(results, epochs, dataset):

    # VALUE PREP & PLOTTING
    experiments = ["baseline", "l2", "dropout", "full"]
    metrics = ["train_loss", "val_acc", "test_acc"]
    colors = {"baseline":"orange", "l2":"blue", "dropout":"green", "full":"red"}

    def to_cpu_numpy(x):
        """Recursively convert tensors on GPU to CPU numpy arrays"""
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        elif isinstance(x, list):
            return [to_cpu_numpy(i) for i in x]
        else:
            return x

    # Convert all results to numpy arrays and compute mean/std
    mean_std_results = {}
    for exp in experiments:
        mean_std_results[exp] = {}
        for metric in metrics:
            data = to_cpu_numpy(results[exp][metric])
            data = np.array(data)
            if metric == "test_acc":
                mean_std_results[exp][metric] = {
                    "mean": data.mean(),
                    "std": data.std()
                }
            else:
                mean_std_results[exp][metric] = {
                    "mean": data.mean(axis=0),
                    "std": data.std(axis=0)
                }

    # Plot Train Loss
    plt.figure(figsize=(8,5))
    for exp in experiments:
        mean = mean_std_results[exp]["train_loss"]["mean"]
        std = mean_std_results[exp]["train_loss"]["std"]
        plt.plot(np.arange(1, epochs+1), mean, label=exp, color=colors[exp])
        plt.fill_between(np.arange(1, epochs+1), mean - std, mean + std, color=colors[exp], alpha=0.2)
    plt.title(f"Training Loss per Epoch for {dataset} dataset")
    plt.xlabel("Epoch")
    plt.ylabel("Cross Entropy Loss")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot Validation Accuracy
    plt.figure(figsize=(8,5))
    for exp in experiments:
        mean = mean_std_results[exp]["val_acc"]["mean"]
        std = mean_std_results[exp]["val_acc"]["std"]
        plt.plot(np.arange(1, epochs+1), mean, label=exp, color=colors[exp])
        plt.fill_between(np.arange(1, epochs+1), mean - std, mean + std, color=colors[exp], alpha=0.2)
    plt.title(f"Validation Accuracy per Epoch for {dataset} dataset")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot Test Accuracy (final value per run)
    plt.figure(figsize=(8,5))
    means = [mean_std_results[exp]["test_acc"]["mean"] for exp in experiments]
    stds  = [mean_std_results[exp]["test_acc"]["std"]  for exp in experiments]
    plt.bar(experiments, means, yerr=stds, color=[colors[exp] for exp in experiments], alpha=0.7, capsize=5)
    plt.title(f"Test Accuracy (Mean ± Std over runs) for {dataset} dataset")
    plt.ylabel("Accuracy")
    plt.ylim(0,1)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()
    
    return