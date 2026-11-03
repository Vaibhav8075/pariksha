"""PyTorch training loop with evaluation-mode and model-selection mistakes (example input)."""
import torch
from torch import nn


def evaluate(model, loader):
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            correct += (model(x).argmax(1) == y).sum().item()
            total += len(y)
    return correct / total


def main(model, train_loader, test_loader, epochs=30):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max")
    loss_fn = nn.CrossEntropyLoss()
    best_acc = 0.0
    model.train()
    for epoch in range(epochs):
        for x, y in train_loader:
            opt.zero_grad()
            loss_fn(model(x), y).backward()
            opt.step()
        model.eval()
        test_acc = evaluate(model, test_loader)
        sched.step(test_acc)
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "best.pt")
    print(f"best test accuracy: {best_acc:.3f}")
