"""Classify a single image as chihuahua or muffin.

Usage:
    python predict.py path/to/image.png
    python predict.py path/to/image.png --model models/chm_cnn.pt

Prints the predicted label and class probabilities, and returns a dict from
`classify()` so it can be imported and tested.
"""
from __future__ import annotations

import os
import json
import argparse
import torch

from chm.model import TinyCNN, CLASSES
from chm.data import load_image_file

ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(ROOT, "models", "chm_cnn.pt")


def load_model(model_path: str = DEFAULT_MODEL):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"model not found at {model_path} — run `python train.py` first")
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    classes = ckpt.get("classes", list(CLASSES))
    model = TinyCNN(len(classes))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, classes


@torch.no_grad()
def classify(image_path: str, model=None, classes=None) -> dict:
    if model is None:
        model, classes = load_model()
    if classes is None:
        classes = list(CLASSES)
    x = load_image_file(image_path).unsqueeze(0)
    probs = torch.softmax(model(x), dim=1)[0].tolist()
    j = int(max(range(len(probs)), key=lambda k: probs[k]))
    return {
        "image": image_path,
        "label": classes[j],
        "prob": round(probs[j], 4),
        "probabilities": {c: round(p, 4) for c, p in zip(classes, probs)},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="path to an image file")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    a = ap.parse_args()
    model, classes = load_model(a.model)
    print(json.dumps(classify(a.image, model, classes), indent=2))


if __name__ == "__main__":
    main()
