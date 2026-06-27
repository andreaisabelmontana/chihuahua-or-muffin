"""Tests for the dataset generator and preprocessing."""
import numpy as np
import torch

from data.generate import make_image, CLASSES as GEN_CLASSES, IMG
from chm.data import preprocess


def test_generator_shapes_and_classes():
    assert tuple(GEN_CLASSES) == ("chihuahua", "muffin")
    for cls in GEN_CLASSES:
        img = make_image(cls, seed=1)
        assert img.shape == (IMG, IMG, 3)
        assert img.dtype == np.uint8


def test_generator_is_deterministic_per_seed():
    a = make_image("muffin", seed=42)
    b = make_image("muffin", seed=42)
    assert np.array_equal(a, b)                       # same seed -> same image


def test_generator_has_intra_class_variation():
    a = make_image("chihuahua", seed=1)
    b = make_image("chihuahua", seed=2)
    assert not np.array_equal(a, b)                   # different seed -> differs


def test_classes_not_separable_by_mean_brightness():
    """The point of the dataset: colour alone must not give it away."""
    ch = np.mean([make_image("chihuahua", s).mean() for s in range(30)])
    mf = np.mean([make_image("muffin", s).mean() for s in range(30)])
    assert abs(ch - mf) < 8.0                         # nearly identical means


def test_preprocess_output():
    img = make_image("muffin", seed=3)
    t = preprocess(img)
    assert t.shape == (3, IMG, IMG)
    assert t.dtype == torch.float32
    assert 0.0 <= float(t.min()) and float(t.max()) <= 1.0
