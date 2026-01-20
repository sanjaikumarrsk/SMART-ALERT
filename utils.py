import os
import torch

def accuracy(output, target):
    with torch.no_grad():
        preds = torch.argmax(output, dim=1)
        correct = (preds == target).float().sum()
        return (correct / target.shape[0]).item()

def save_checkpoint(state, is_best, out_dir, filename="last_model.pth"):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    torch.save(state, path)
    if is_best:
        best_path = os.path.join(out_dir, "best_model.pth")
        torch.save(state, best_path)

