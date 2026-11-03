"""PyTorch loop with a separate validation set, eval mode and a seed: no findings."""
import torch
from torch import nn

torch.manual_seed(0)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    for x, y in loader:
        correct += (model(x).argmax(1) == y).sum().item()
        total += len(y)
    return correct / total


def main(model, train_loader, val_loader, test_loader, epochs=30):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max")
    loss_fn = nn.CrossEntropyLoss()
    best_val = 0.0
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            opt.zero_grad()
            loss_fn(model(x), y).backward()
            opt.step()
        val_acc = evaluate(model, val_loader)
        sched.step(val_acc)
        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), "best.pt")
    model.load_state_dict(torch.load("best.pt"))
    print(f"test accuracy (reported once): {evaluate(model, test_loader):.3f}")
