import random

from autograd.value import Value
from nn.scalar_mlp import MLP
from tests.utils import check_model_grads

X = [[0.5, -1.0], [1.5, 0.3], [-0.7, 0.8]]
Y = [1.0, -1.0, 1.0]


def make_loss(model):
    def loss_fn():
        preds = [model(x) for x in X]
        return sum(((p - t) ** 2 for p, t in zip(preds, Y)), Value(0.0))
    return loss_fn


def test_parameter_count():
    random.seed(0)
    model = MLP(3, [4, 4, 1])
    # (3*4 + 4) + (4*4 + 4) + (4*1 + 1) = 41
    assert len(model.parameters()) == 41


def test_mlp_gradients_match_numerical():
    random.seed(0)
    model = MLP(2, [3, 3, 1], activation="tanh")
    check_model_grads(model, make_loss(model))


def test_zero_grad_resets_gradients():
    random.seed(0)
    model = MLP(2, [3, 1])
    make_loss(model)().backward()
    assert any(p.grad != 0.0 for p in model.parameters())
    model.zero_grad()
    assert all(p.grad == 0.0 for p in model.parameters())


def test_one_gradient_step_reduces_loss():
    random.seed(0)
    model = MLP(2, [3, 3, 1])
    loss_fn = make_loss(model)
    before = loss_fn()
    model.zero_grad()
    before.backward()
    for p in model.parameters():
        p.data -= 0.01 * p.grad
    assert loss_fn().data < before.data
