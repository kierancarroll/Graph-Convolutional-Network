import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def visualize_embeddings(model, x, y, test_mask, A_hat_torch, dataset):
  model.eval()
  with torch.no_grad():
    logits, embeddings = model(x, A_hat_torch, return_embeddings = True)

    pred = logits.argmax(dim=1)
    correct_test = (pred[test_mask] == y[test_mask]).sum().item()
    accuracy_test = correct_test / test_mask.sum().item()

    print(f"test accuracy = {accuracy_test}")

    # Convert to numpy
    embeddings = embeddings.cpu().numpy()
    labels = y.cpu().numpy()
    mask = test_mask.cpu().numpy()

    # Apply mask BEFORE t-SNE
    embeddings = embeddings[mask]
    labels = labels[mask]

    tsne = TSNE(n_components = 2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    plt.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=labels,
        cmap="tab10",
        s=10
    )
    plt.title(f"t-SNE of GCN Embeddings L2 Model on {dataset} dataset")
    plt.show()