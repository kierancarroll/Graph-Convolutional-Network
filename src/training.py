import torch


def train(model, optimizer, criterion, epochs, x, y, train_mask, val_mask, A_hat_torch):
  
  train_loss, val_acc = [],[]

  for i in range(epochs):
    # train model
    model.train()
    optimizer.zero_grad()
    out_train = model(x, A_hat_torch)
    loss_train = criterion(out_train[train_mask],y[train_mask])
    loss_train.backward()
    optimizer.step()

    # evaluate model
    model.eval()
    with torch.no_grad():
      out_val = model(x, A_hat_torch)
      pred = out_val.argmax(dim=1)
      correct_val = (pred[val_mask] == y[val_mask]).sum().item()
      accuracy_val = correct_val/val_mask.sum().item()

    train_loss.append(loss_train.item())
    val_acc.append(accuracy_val)
    print(f"Epoch: {i+1} | Train Loss: {loss_train} | Val Accuracy: {accuracy_val}")

  return train_loss, val_acc





