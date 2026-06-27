"""End-to-end test: the model learns to beat chance on a held-out split,
and predict.py returns a valid label + probability.

Trains a small model on a small freshly-generated dataset inside a temp dir,
so the test is self-contained and fast (a few seconds on CPU).
"""
import os
import tempfile

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.generate import build
from chm.model import TinyCNN, CLASSES
from chm.data import ImageFolderDataset
import predict as predict_mod


def _train_quick(root, epochs=8):
    torch.manual_seed(0)
    np.random.seed(0)
    train_ds = ImageFolderDataset(os.path.join(root, "train"), augment=True)
    test_ds = ImageFolderDataset(os.path.join(root, "test"))
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=32)
    model = TinyCNN(len(CLASSES))
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
    crit = torch.nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for x, y in train_dl:
            opt.zero_grad()
            crit(model(x), y).backward()
            opt.step()
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in test_dl:
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.numel()
    return model, test_ds, correct / total


def test_model_beats_chance_on_held_out():
    with tempfile.TemporaryDirectory() as d:
        build(d, n_per_class=160, seed=7)
        model, test_ds, acc = _train_quick(d, epochs=8)
        # chance is 0.5; require a clear margin, not a coin flip
        assert acc > 0.7, f"held-out accuracy {acc:.3f} did not beat chance"


def test_predict_returns_valid_label_and_prob():
    with tempfile.TemporaryDirectory() as d:
        build(d, n_per_class=160, seed=7)
        model, test_ds, acc = _train_quick(d, epochs=8)
        # grab one test image path and classify it through predict.classify
        sample_path = test_ds.samples[0][0]
        out = predict_mod.classify(sample_path, model=model, classes=list(CLASSES))
        assert out["label"] in CLASSES
        assert 0.0 <= out["prob"] <= 1.0
        # probabilities form a valid distribution over the two classes
        ps = out["probabilities"]
        assert set(ps.keys()) == set(CLASSES)
        assert abs(sum(ps.values()) - 1.0) < 1e-3
