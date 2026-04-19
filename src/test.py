import torch


def test(model, x, y, test_mask, A_hat_torch):
  model.eval()
  with torch.no_grad():
    out_test = model(x, A_hat_torch)
    pred = out_test.argmax(dim=1)
    correct_test = (pred[test_mask] == y[test_mask]).sum().item()
    accuracy_test = correct_test / test_mask.sum().item()
    return accuracy_test