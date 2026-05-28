import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_model(model, train_loader: DataLoader, val_loader: DataLoader,
                *, epochs: int, lr: float, optimizer: str = "rmsprop",
                device: str = "mps") -> dict:
    model = model.to(device)
    opt = (torch.optim.RMSprop(model.parameters(), lr=lr)
           if optimizer == "rmsprop"
           else torch.optim.Adam(model.parameters(), lr=lr))
    loss_fn = nn.CrossEntropyLoss()
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    for ep in range(epochs):
        model.train()
        tloss = 0.0
        n = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
            tloss += loss.item() * xb.size(0)
            n += xb.size(0)
        history["train_loss"].append(tloss / n)
        model.eval()
        vloss = 0.0
        vcorrect = 0
        vn = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                loss = loss_fn(out, yb)
                vloss += loss.item() * xb.size(0)
                vcorrect += (out.argmax(1) == yb).sum().item()
                vn += xb.size(0)
        history["val_loss"].append(vloss / vn)
        history["val_acc"].append(vcorrect / vn)
        print(f"ep {ep+1:>3}  train_loss={history['train_loss'][-1]:.4f}  "
              f"val_loss={history['val_loss'][-1]:.4f}  val_acc={history['val_acc'][-1]:.4f}")
    return history
