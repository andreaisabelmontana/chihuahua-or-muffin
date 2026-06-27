"""Numerical gradient check on a tiny conv + dense stack.

Confirms autograd's analytic gradient matches a finite-difference estimate,
i.e. the backprop through conv/relu/linear is correct. Uses float64 for a
tight tolerance via torch.autograd.gradcheck.
"""
import torch


def _tiny_net(x, w_conv, w_fc, b_fc):
    # conv (no padding): 1x1x5x5 -> 2x3x3, relu, flatten, linear -> 2 logits
    z = torch.nn.functional.conv2d(x, w_conv)
    z = torch.relu(z)
    z = z.flatten(1)
    return z @ w_fc.t() + b_fc


def test_conv_dense_gradcheck():
    torch.manual_seed(0)
    x = torch.randn(2, 1, 5, 5, dtype=torch.double, requires_grad=True)
    w_conv = torch.randn(2, 1, 3, 3, dtype=torch.double, requires_grad=True)
    w_fc = torch.randn(2, 2 * 3 * 3, dtype=torch.double, requires_grad=True)
    b_fc = torch.randn(2, dtype=torch.double, requires_grad=True)

    assert torch.autograd.gradcheck(
        _tiny_net, (x, w_conv, w_fc, b_fc), eps=1e-6, atol=1e-4
    )


def test_relu_gradcheck():
    x = torch.randn(4, 4, dtype=torch.double, requires_grad=True) + 0.5
    assert torch.autograd.gradcheck(torch.relu, (x,), eps=1e-6, atol=1e-4)
