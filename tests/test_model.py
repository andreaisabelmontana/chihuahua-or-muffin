"""Forward-shape and gradient-flow tests for TinyCNN and its conv ops."""
import torch

from chm.model import TinyCNN, CLASSES


def test_classes():
    assert tuple(CLASSES) == ("chihuahua", "muffin")


def test_forward_shapes():
    model = TinyCNN(len(CLASSES))
    x = torch.randn(5, 3, 64, 64)
    out = model(x)
    assert out.shape == (5, len(CLASSES))           # batch x 2 logits


def test_feature_map_shapes():
    """Each conv block halves spatial dims and grows channels as designed."""
    model = TinyCNN()
    x = torch.randn(2, 3, 64, 64)
    f = x
    expected = [(2, 16, 32, 32), (2, 32, 16, 16), (2, 64, 8, 8)]
    for blk, exp in zip(model.features, expected):
        f = blk(f)
        assert tuple(f.shape) == exp
    pooled = model.pool(f)
    assert tuple(pooled.shape) == (2, 64, 1, 1)      # global average pool


def test_single_conv_op_shape():
    """A bare 3x3 same-padding conv keeps spatial size and sets channels."""
    conv = torch.nn.Conv2d(3, 8, kernel_size=3, padding=1)
    y = conv(torch.randn(1, 3, 64, 64))
    assert tuple(y.shape) == (1, 8, 64, 64)


def test_gradients_flow():
    """A backward pass produces finite, non-trivial gradients on every param."""
    model = TinyCNN()
    x = torch.randn(4, 3, 64, 64)
    y = torch.randint(0, 2, (4,))
    loss = torch.nn.functional.cross_entropy(model(x), y)
    loss.backward()
    total = 0.0
    for p in model.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()
        total += p.grad.abs().sum().item()
    assert total > 0.0


def test_num_params_reasonable():
    n = TinyCNN().num_params()
    assert 10_000 < n < 200_000
