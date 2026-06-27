"""Train the chihuahua-or-muffin CNN end to end.

Steps
  1. If data/dataset is missing, generate it (data/generate.py).
  2. Train TinyCNN with Adam + cross-entropy on the train split,
     model-select on the val split.
  3. Evaluate the best model on the held-out TEST split.
  4. Save:  models/chm_cnn.pt          (weights + meta)
            figures/training_curve.png (loss + val/test accuracy)
            figures/predictions.png    (grid of test images w/ pred & prob)
            results.json               (REAL held-out metrics)

Run:  python train.py            (defaults are quick on CPU)
"""
from __future__ import annotations

import os
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from chm.model import TinyCNN, CLASSES
from chm.data import ImageFolderDataset, dataset_root

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(ROOT, "models")
FIG_DIR = os.path.join(ROOT, "figures")


def ensure_data(n_per_class: int, seed: int):
    root = dataset_root()
    if os.path.isdir(os.path.join(root, "train", CLASSES[0])):
        return root
    from data.generate import build
    build(root, n_per_class=n_per_class, seed=seed)
    return root


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / total


@torch.no_grad()
def confusion(model, loader, device, n=2):
    model.eval()
    cm = np.zeros((n, n), dtype=int)
    for x, y in loader:
        pred = model(x.to(device)).argmax(1).cpu().numpy()
        for t, p in zip(y.numpy(), pred):
            cm[t, p] += 1
    return cm


def save_training_curve(hist, path):
    ep = range(1, len(hist["train_loss"]) + 1)
    fig, ax1 = plt.subplots(figsize=(7, 4.2))
    ax1.plot(ep, hist["train_loss"], color="#b45309", label="train loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("cross-entropy loss", color="#b45309")
    ax1.tick_params(axis="y", labelcolor="#b45309")
    ax2 = ax1.twinx()
    ax2.plot(ep, hist["val_acc"], color="#1d4ed8", marker="o",
             ms=3, label="val accuracy")
    ax2.set_ylabel("validation accuracy", color="#1d4ed8")
    ax2.tick_params(axis="y", labelcolor="#1d4ed8")
    ax2.set_ylim(0.4, 1.02)
    fig.suptitle("chihuahua-or-muffin — training curve")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


@torch.no_grad()
def save_prediction_grid(model, test_ds, device, path, cols=6, rows=4):
    model.eval()
    idx = np.random.RandomState(0).choice(len(test_ds),
                                          size=min(cols * rows, len(test_ds)),
                                          replace=False)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.7, rows * 1.9))
    for ax, i in zip(axes.ravel(), idx):
        x, y = test_ds[i]
        logits = model(x.unsqueeze(0).to(device))
        prob = torch.softmax(logits, 1)[0].cpu().numpy()
        pj = int(prob.argmax())
        img = x.permute(1, 2, 0).numpy()
        ax.imshow(np.clip(img, 0, 1))
        ax.axis("off")
        ok = (pj == y)
        ax.set_title(f"{CLASSES[pj]}\n{prob[pj]*100:.0f}%",
                     color=("#15803d" if ok else "#b91c1c"), fontsize=8)
    # blank any unused axes
    for ax in axes.ravel()[len(idx):]:
        ax.axis("off")
    fig.suptitle("Test predictions (green = correct, red = wrong)", y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cpu")

    root = ensure_data(args.n_per_class, args.seed)
    train_ds = ImageFolderDataset(os.path.join(root, "train"), augment=True)
    val_ds = ImageFolderDataset(os.path.join(root, "val"))
    test_ds = ImageFolderDataset(os.path.join(root, "test"))

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch)
    test_dl = DataLoader(test_ds, batch_size=args.batch)

    model = TinyCNN(len(CLASSES)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()

    print(f"Model params: {model.num_params():,}")
    print(f"Splits  train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

    hist = {"train_loss": [], "val_acc": []}
    best_val, best_state = -1.0, None
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)
        tl = running / len(train_ds)
        va = evaluate(model, val_dl, device)
        hist["train_loss"].append(tl)
        hist["val_acc"].append(va)
        if va > best_val:
            best_val = va
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f"epoch {ep:2d}/{args.epochs}  loss {tl:.4f}  val_acc {va:.3f}")

    # restore best-on-val model, then evaluate on the untouched TEST split
    model.load_state_dict(best_state)
    test_acc = evaluate(model, test_dl, device)
    cm = confusion(model, test_dl, device, n=len(CLASSES))
    elapsed = time.time() - t0

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "classes": list(CLASSES),
                "arch": "TinyCNN"},
               os.path.join(MODELS_DIR, "chm_cnn.pt"))
    save_training_curve(hist, os.path.join(FIG_DIR, "training_curve.png"))
    save_prediction_grid(model, test_ds, device,
                         os.path.join(FIG_DIR, "predictions.png"))

    results = {
        "model": "TinyCNN (3 conv blocks + GAP + dense head)",
        "params": model.num_params(),
        "framework": f"pytorch {torch.__version__}",
        "dataset": {
            "kind": "procedurally rendered (numpy+OpenCV), NOT the real meme photos",
            "classes": list(CLASSES),
            "image_size": 64,
            "n_train": len(train_ds),
            "n_val": len(val_ds),
            "n_test": len(test_ds),
        },
        "epochs": args.epochs,
        "best_val_accuracy": round(best_val, 4),
        "test_accuracy": round(test_acc, 4),
        "chance_baseline": 0.5,
        "confusion_matrix": {
            "labels": list(CLASSES),
            "rows_true_cols_pred": cm.tolist(),
        },
        "train_seconds": round(elapsed, 1),
    }
    with open(os.path.join(ROOT, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== RESULTS ===")
    print(f"best val accuracy : {best_val:.3f}")
    print(f"HELD-OUT TEST acc : {test_acc:.3f}   (chance = 0.500)")
    print(f"confusion (rows=true {CLASSES}, cols=pred):\n{cm}")
    print(f"trained in {elapsed:.1f}s  ->  results.json, models/, figures/")
    return results


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--n-per-class", type=int, default=600, dest="n_per_class")
    ap.add_argument("--seed", type=int, default=0)
    return ap


if __name__ == "__main__":
    train(build_argparser().parse_args())
