"""Day 4 tests: my layers (nn/layers.py) and losses (nn/losses.py).

Three kinds of proof, so that one weak spot in one kind is covered by another:
  1. comparison with PyTorch in float64            (does my code do the same math as a trusted library?)
  2. numerical gradient check (no PyTorch needed)   (is my backward the true slope of my forward?)
  3. hand-computed / property tests                 (small facts I can check on paper)
Then I break my own code on purpose (bug injection) to see if these tests notice.
"""
import math

import numpy as np
import pytest

from autograd.tensor import Tensor
from nn import layers, losses
from tests import layer_report as report
from tests.tensor_utils import scaled_error

try:                                   # PyTorch is only used to CHECK my work, never inside my engine
    import torch
    import torch.nn.functional as F
except ImportError:                    # if it is missing, only the PyTorch tests are skipped
    torch = None

needs_torch = pytest.mark.skipif(torch is None, reason="torch is not installed")
TOL = report.TOL                       # 1e-6, from the checklist
LN10 = math.log(10)


# =====================================================================================
# Helpers
# =====================================================================================

def compare_with_torch(component, my_fn, torch_fn, arrays, names):
    """Run the same computation in my engine and in PyTorch, compare the output and every gradient."""
    arrays = [np.array(a, dtype=float) for a in arrays]

    # --- my side ---
    mine = [Tensor(a) for a in arrays]                   # my Tensors holding the inputs
    my_out = my_fn(*mine)                                # my forward pass

    # --- PyTorch side (float64, same numbers) ---
    th = [torch.tensor(a, dtype=torch.float64, requires_grad=True) for a in arrays]
    th_out = torch_fn(*th)                               # PyTorch forward pass

    forward_err = scaled_error(my_out.data, th_out.detach().numpy())     # compare the outputs

    # A random "upstream" weight for every output value, so each output element has a different gradient
    # (an all-ones upstream gradient can hide mistakes).
    weights = np.random.default_rng(7).normal(size=my_out.data.shape)
    (my_out * Tensor(weights)).sum().backward()                         # my backward pass
    (th_out * torch.tensor(weights)).sum().backward()                   # PyTorch backward pass

    grad_errs = {n: scaled_error(m.grad, t.grad.numpy()) for n, m, t in zip(names, mine, th)}
    report.record_torch(component, forward_err, grad_errs)
    return forward_err, grad_errs


def numerical_check(loss_fn, tensors):
    """Compare backward() with the slope measured by nudging every number in every tensor.

    loss_fn(): builds the forward pass from the CURRENT .data of `tensors` and returns a scalar Tensor.
    tensors  : the inputs and weights I want gradients for.
    Returns the worst scaled error.
    """
    eps = 1e-6
    for t in tensors:
        t.grad = np.zeros_like(t.data)                    # start from zero gradients
    loss_fn().backward()                                  # my analytic gradients
    analytic = [t.grad.copy() for t in tensors]           # keep a copy, because forward calls below don't touch .grad

    worst = 0.0
    for t, a in zip(tensors, analytic):
        num = np.zeros_like(t.data)
        it = np.nditer(t.data, flags=["multi_index"])
        for _ in it:
            idx = it.multi_index
            old = t.data[idx]
            t.data[idx] = old + eps
            plus = float(loss_fn().data)                  # loss with this one number nudged up
            t.data[idx] = old - eps
            minus = float(loss_fn().data)                 # loss with it nudged down
            t.data[idx] = old                             # ALWAYS put it back
            num[idx] = (plus - minus) / (2 * eps)         # central difference = the true slope
        worst = max(worst, scaled_error(a, num))
    return worst


def make_net(rng, sizes, last_gain=1.0, act=layers.ReLU):
    """Linear -> act -> Linear -> act -> ... -> Linear. sizes = [in, h1, h2, ..., out]."""
    parts = []
    for i in range(len(sizes) - 1):
        is_last = i == len(sizes) - 2
        parts.append(layers.Linear(sizes[i], sizes[i + 1], rng, gain=last_gain if is_last else 1.0))
        if not is_last:
            parts.append(act())
    return layers.Sequential(*parts)


# =====================================================================================
# 1. COMPARISON WITH PYTORCH
# =====================================================================================

rng0 = np.random.default_rng(0)
X = rng0.normal(size=(6, 5))          # 6 examples, 5 features
Wm = rng0.normal(size=(5, 4))         # weight of shape (in, out)
bv = rng0.normal(size=4)
SHIFT = rng0.normal(size=(6, 4)) + 0.05          # keeps values away from exactly 0 (the ReLU kink)
LOGITS = rng0.normal(size=(8, 10)) * 2.0
LABELS = rng0.integers(0, 10, size=8)
PRED, TARGET = rng0.normal(size=(7, 3)), rng0.normal(size=(7, 3))


def _my_linear(x, W, b):
    layer = layers.Linear(W.shape[0], W.shape[1], np.random.default_rng(0))
    layer.W, layer.b = W, b                          # use the exact weights that PyTorch gets
    return layer(x)


@needs_torch
def test_linear_matches_torch():
    # torch.nn.functional.linear(x, weight, bias) = x @ weight.T + bias, and torch stores weight as (out, in),
    # so I give it W.T to get the same thing as my (in, out) layout.
    f, g = compare_with_torch("Linear (in=5, out=4)", _my_linear,
                              lambda x, W, b: F.linear(x, W.T, b), [X, Wm, bv], ["x", "W", "b"])
    assert f < TOL and max(g.values()) < TOL


@needs_torch
@pytest.mark.parametrize("name,mine,theirs", [
    ("ReLU", lambda x: layers.ReLU()(x), torch.relu if torch else None),
    ("Tanh", lambda x: layers.Tanh()(x), torch.tanh if torch else None),
    ("Sigmoid", lambda x: layers.Sigmoid()(x), torch.sigmoid if torch else None),
])
def test_activation_matches_torch(name, mine, theirs):
    f, g = compare_with_torch(name, mine, theirs, [SHIFT], ["x"])
    assert f < TOL and max(g.values()) < TOL


@needs_torch
def test_sigmoid_matches_torch_on_extreme_inputs():
    # Large positive and negative numbers are where a careless sigmoid overflows.
    extreme = np.array([[-800.0, -50.0, -1.0, 0.0, 1.0, 50.0, 800.0]])
    with np.errstate(over="raise", invalid="raise"):      # an overflow or a nan becomes an error (a harmless underflow to 0 is fine)
        f, g = compare_with_torch("Sigmoid (extreme inputs)", lambda x: layers.Sigmoid()(x),
                                  torch.sigmoid, [extreme], ["x"])
    assert f < TOL and max(g.values()) < TOL


@needs_torch
def test_mse_loss_matches_torch():
    f, g = compare_with_torch("mse_loss", losses.mse_loss, F.mse_loss, [PRED, TARGET], ["pred", "target"])
    assert f < TOL and max(g.values()) < TOL


@needs_torch
def test_softmax_cross_entropy_matches_torch():
    f, g = compare_with_torch("softmax_cross_entropy",
                              lambda z: losses.softmax_cross_entropy(z, LABELS),
                              lambda z: F.cross_entropy(z, torch.tensor(LABELS)), [LOGITS], ["logits"])
    assert f < TOL and max(g.values()) < TOL


@needs_torch
def test_softmax_cross_entropy_matches_torch_with_huge_logits():
    big = LOGITS * 400.0                                  # logits in the hundreds/thousands: naive exp() would overflow
    f, g = compare_with_torch("softmax_cross_entropy (huge logits)",
                              lambda z: losses.softmax_cross_entropy(z, LABELS),
                              lambda z: F.cross_entropy(z, torch.tensor(LABELS)), [big], ["logits"])
    assert f < TOL and max(g.values()) < TOL


@needs_torch
def test_whole_network_matches_torch_model():
    """The checklist item: copy MY weights into an equivalent torch model and compare everything."""
    rng = np.random.default_rng(3)
    net = make_net(rng, [6, 8, 5, 3], act=layers.Tanh)            # Linear-Tanh-Linear-Tanh-Linear
    x = rng.normal(size=(9, 6))
    y = rng.integers(0, 3, size=9)

    # Build the same architecture in torch, then copy my numbers into it.
    tmodel = torch.nn.Sequential(torch.nn.Linear(6, 8), torch.nn.Tanh(), torch.nn.Linear(8, 5),
                                 torch.nn.Tanh(), torch.nn.Linear(5, 3)).double()
    my_linears = [l for l in net.layers if isinstance(l, layers.Linear)]
    t_linears = [l for l in tmodel if isinstance(l, torch.nn.Linear)]
    for mine, theirs in zip(my_linears, t_linears):
        theirs.weight.data = torch.tensor(mine.W.data.T.copy())    # torch weight is (out, in): transpose mine
        theirs.bias.data = torch.tensor(mine.b.data.copy())

    # forward + loss + backward on both sides
    net.zero_grad()
    my_logits = net(Tensor(x))
    my_loss = losses.softmax_cross_entropy(my_logits, y)
    my_loss.backward()
    t_logits = tmodel(torch.tensor(x))
    t_loss = F.cross_entropy(t_logits, torch.tensor(y))
    t_loss.backward()

    fwd = max(scaled_error(my_logits.data, t_logits.detach().numpy()),
              scaled_error(my_loss.data, t_loss.item()))
    grads = {}
    for i, (mine, theirs) in enumerate(zip(my_linears, t_linears), start=1):
        grads[f"W{i}"] = scaled_error(mine.W.grad, theirs.weight.grad.numpy().T)   # transpose back
        grads[f"b{i}"] = scaled_error(mine.b.grad, theirs.bias.grad.numpy())
    report.record_torch("3-layer net (Tanh) + cross-entropy", fwd, grads)
    assert fwd < TOL and max(grads.values()) < TOL


@needs_torch
def test_relu_network_matches_torch_model():
    rng = np.random.default_rng(4)
    net = make_net(rng, [10, 16, 16, 10], act=layers.ReLU)
    x = rng.normal(size=(12, 10))
    y = rng.integers(0, 10, size=12)
    tmodel = torch.nn.Sequential(torch.nn.Linear(10, 16), torch.nn.ReLU(), torch.nn.Linear(16, 16),
                                 torch.nn.ReLU(), torch.nn.Linear(16, 10)).double()
    my_linears = [l for l in net.layers if isinstance(l, layers.Linear)]
    t_linears = [l for l in tmodel if isinstance(l, torch.nn.Linear)]
    for mine, theirs in zip(my_linears, t_linears):
        theirs.weight.data = torch.tensor(mine.W.data.T.copy())
        theirs.bias.data = torch.tensor(mine.b.data.copy())
    net.zero_grad()
    my_loss = losses.softmax_cross_entropy(net(Tensor(x)), y)
    my_loss.backward()
    t_loss = F.cross_entropy(tmodel(torch.tensor(x)), torch.tensor(y))
    t_loss.backward()
    fwd = scaled_error(my_loss.data, t_loss.item())
    grads = {}
    for i, (mine, theirs) in enumerate(zip(my_linears, t_linears), start=1):
        grads[f"W{i}"] = scaled_error(mine.W.grad, theirs.weight.grad.numpy().T)
        grads[f"b{i}"] = scaled_error(mine.b.grad, theirs.bias.grad.numpy())
    report.record_torch("3-layer net (ReLU) + cross-entropy", fwd, grads)
    assert fwd < TOL and max(grads.values()) < TOL


# =====================================================================================
# 2. NUMERICAL GRADIENT CHECKS (independent of PyTorch)
# =====================================================================================

def _case_linear():
    x, W, b = Tensor(X.copy()), Tensor(Wm.copy()), Tensor(bv.copy())
    return (lambda: _my_linear(x, W, b).tanh().mean()), [x, W, b]


def _case_activation(layer_cls):
    def build():
        x = Tensor(SHIFT.copy())
        return (lambda: layer_cls()(x).sum() * 1.0), [x]
    return build


def _case_mse():
    p, t = Tensor(PRED.copy()), Tensor(TARGET.copy())
    return (lambda: losses.mse_loss(p, t)), [p, t]


def _case_ce(scale=1.0):
    def build():
        z = Tensor(LOGITS.copy() * scale)
        return (lambda: losses.softmax_cross_entropy(z, LABELS)), [z]
    return build


def _case_net(act):
    def build():
        rng = np.random.default_rng(5)
        net = make_net(rng, [4, 6, 3], act=act)
        x = Tensor(rng.normal(size=(5, 4)))
        y = rng.integers(0, 3, size=5)
        return (lambda: losses.softmax_cross_entropy(net(x), y)), [x] + net.parameters()
    return build


NUMERIC_CASES = [
    ("Linear -> tanh -> mean", _case_linear),
    ("ReLU", lambda: (lambda x: ((lambda: layers.ReLU()(x).sum()), [x]))(Tensor(SHIFT.copy()))),
    ("Tanh", _case_activation(layers.Tanh)),
    ("Sigmoid", _case_activation(layers.Sigmoid)),
    ("mse_loss", _case_mse),
    ("softmax_cross_entropy", _case_ce(1.0)),
    ("softmax_cross_entropy (logits x 5)", _case_ce(5.0)),
    ("net Linear-ReLU-Linear + CE (all weights)", _case_net(layers.ReLU)),
    ("net Linear-Tanh-Linear + CE (all weights)", _case_net(layers.Tanh)),
    ("net Linear-Sigmoid-Linear + CE (all weights)", _case_net(layers.Sigmoid)),
]


@pytest.mark.parametrize("name,build", NUMERIC_CASES, ids=[c[0] for c in NUMERIC_CASES])
def test_numerical_gradients(name, build):
    loss_fn, tensors = build()
    worst = numerical_check(loss_fn, tensors)
    report.record_numeric(name, worst, worst < TOL)
    assert worst < TOL, f"{name}: my backward() disagrees with the numerical slope (scaled error {worst:.2e})"


# =====================================================================================
# 3. HAND-COMPUTED AND PROPERTY TESTS
# =====================================================================================

def test_cross_entropy_by_hand():
    # logits [1,2,3], correct class 2: loss = -log(e^3 / (e^1 + e^2 + e^3)) = log(1 + e^-1 + e^-2) = 0.40760596...
    loss = losses.softmax_cross_entropy(Tensor([[1.0, 2.0, 3.0]]), np.array([2]))
    assert abs(float(loss.data) - math.log(1 + math.exp(-1) + math.exp(-2))) < 1e-12


def test_equal_logits_give_ln_of_number_of_classes():
    for k in (2, 5, 10):
        loss = losses.softmax_cross_entropy(Tensor(np.zeros((4, k))), np.zeros(4, dtype=int))
        assert abs(float(loss.data) - math.log(k)) < 1e-12      # uniform guess -> loss = ln(K)


def test_cross_entropy_gradient_rows_sum_to_zero():
    # (p - onehot)/N: each row of p adds to 1 and each onehot row adds to 1, so every row of the gradient adds to 0.
    z = Tensor(LOGITS.copy())
    losses.softmax_cross_entropy(z, LABELS).backward()
    assert np.allclose(z.grad.sum(axis=1), 0.0, atol=1e-15)


def test_cross_entropy_gradient_signs():
    # My derivation: the correct class gets a NEGATIVE gradient (its score is pushed up), all others POSITIVE.
    z = Tensor(LOGITS.copy())
    losses.softmax_cross_entropy(z, LABELS).backward()
    rows = np.arange(len(LABELS))
    assert (z.grad[rows, LABELS] < 0).all()
    mask = np.ones_like(z.grad, dtype=bool)
    mask[rows, LABELS] = False
    assert (z.grad[mask] > 0).all()


def test_cross_entropy_gradient_is_divided_by_batch_size():
    z1 = Tensor(np.tile(LOGITS[:1], (1, 1)).copy())
    z4 = Tensor(np.tile(LOGITS[:1], (4, 1)).copy())              # the same example repeated 4 times
    losses.softmax_cross_entropy(z1, LABELS[:1]).backward()
    losses.softmax_cross_entropy(z4, np.repeat(LABELS[:1], 4)).backward()
    assert np.allclose(z4.grad[0], z1.grad[0] / 4)               # the average over 4 copies shrinks each share by 4


def test_cross_entropy_is_stable_for_huge_logits():
    z = Tensor(np.array([[1000.0, 1001.0, 1002.0]]))
    loss = losses.softmax_cross_entropy(z, np.array([2]))
    loss.backward()
    assert np.isfinite(loss.data) and np.isfinite(z.grad).all()
    assert abs(float(loss.data) - math.log(1 + math.exp(-1) + math.exp(-2))) < 1e-9


def test_cross_entropy_does_not_change_when_every_logit_is_shifted():
    # softmax(z + c) = softmax(z), which is exactly why subtracting the max is allowed.
    a = losses.softmax_cross_entropy(Tensor(LOGITS.copy()), LABELS)
    b = losses.softmax_cross_entropy(Tensor(LOGITS.copy() + 123.0), LABELS)
    assert abs(float(a.data) - float(b.data)) < 1e-9


def test_cross_entropy_rejects_bad_labels():
    with pytest.raises(AssertionError):
        losses.softmax_cross_entropy(Tensor(np.zeros((2, 3))), np.array([0, 3]))   # class 3 does not exist
    with pytest.raises(AssertionError):
        losses.softmax_cross_entropy(Tensor(np.zeros((2, 3))), np.array([0]))       # wrong number of labels


def test_mse_by_hand():
    assert abs(float(losses.mse_loss(Tensor([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 5.0])).data) - 4.0 / 3.0) < 1e-12


def test_linear_shapes_and_initial_bias():
    layer = layers.Linear(7, 3, np.random.default_rng(0))
    assert layer.W.data.shape == (7, 3) and layer.b.data.shape == (3,)     # weight is (in, out)
    assert not layer.b.data.any()                                          # bias starts at zero
    assert layer(Tensor(np.ones((5, 7)))).data.shape == (5, 3)


def test_he_initialization_has_the_right_spread():
    layer = layers.Linear(2000, 500, np.random.default_rng(0))
    expected = math.sqrt(2.0 / 2000)
    measured = float(layer.W.data.std())
    report.record_signal("He weights: measured std / sqrt(2/n)", measured / expected, "should be about 1.0")
    assert abs(measured / expected - 1.0) < 0.03


def _signal_rms(gain, depth=10, width=256, seed=0):
    rng = np.random.default_rng(seed)
    x = Tensor(rng.normal(size=(512, width)))
    for _ in range(depth):
        x = layers.ReLU()(layers.Linear(width, width, rng, gain=gain)(x))
    return float(np.sqrt((x.data ** 2).mean()))


def test_he_init_keeps_the_signal_alive_through_ten_relu_layers():
    he = _signal_rms(1.0)
    xavier = _signal_rms(1.0 / math.sqrt(2.0))             # Xavier is He without the factor 2
    tiny = _signal_rms(0.01 / math.sqrt(2.0 / 256))        # weights of std 0.01
    big = _signal_rms(1.0 / math.sqrt(2.0 / 256))          # weights of std 1.0
    report.record_signal("tiny weights (std 0.01)", tiny, "signal vanished")
    report.record_signal("Xavier sqrt(1/n) with ReLU", xavier, "shrinks: ReLU zeroes half of it")
    report.record_signal("He sqrt(2/n) with ReLU", he, "stays about 1: healthy")
    report.record_signal("big weights (std 1.0)", big, "signal exploded")
    assert 0.5 < he < 2.0
    assert xavier < 0.5 * he
    assert tiny < 1e-3 and big > 1e3


def test_initial_loss_of_a_fresh_10_class_network_is_close_to_ln10():
    losses_default, losses_small = [], []
    for seed in range(60):                                            # many seeds so one lucky draw can't fool me
        rng = np.random.default_rng(seed)
        x = Tensor(rng.normal(size=(128, 20)))                        # random data
        y = rng.integers(0, 10, size=128)                             # random labels
        for gain, store in ((1.0, losses_default), (0.01, losses_small)):
            net = make_net(np.random.default_rng(seed + 1000), [20, 64, 64, 10], last_gain=gain)
            store.append(float(losses.softmax_cross_entropy(net(x), y).data))
    report.record_init_loss("He init everywhere (last layer gain 1.0)", np.mean(losses_default), np.std(losses_default), 60)
    report.record_init_loss("He init, small last layer (gain 0.01)", np.mean(losses_small), np.std(losses_small), 60)
    assert abs(np.mean(losses_small) - LN10) < 0.05, "a fresh network should start near ln(10)"
    assert np.mean(losses_default) > LN10                          # the unscaled last layer starts higher: a useful fact


def test_zero_grad_clears_every_parameter():
    rng = np.random.default_rng(0)
    net = make_net(rng, [4, 5, 3])
    losses.softmax_cross_entropy(net(Tensor(rng.normal(size=(6, 4)))), rng.integers(0, 3, size=6)).backward()
    assert any(p.grad.any() for p in net.parameters())             # gradients exist after backward
    net.zero_grad()
    assert not any(p.grad.any() for p in net.parameters())         # and are all zero after zero_grad


def test_gradients_accumulate_without_zero_grad():
    rng = np.random.default_rng(0)
    net = make_net(rng, [4, 5, 3])
    x, y = Tensor(rng.normal(size=(6, 4))), rng.integers(0, 3, size=6)
    losses.softmax_cross_entropy(net(x), y).backward()
    once = net.parameters()[0].grad.copy()
    losses.softmax_cross_entropy(net(x), y).backward()             # second backward WITHOUT zero_grad
    assert np.allclose(net.parameters()[0].grad, 2 * once)         # += doubles it: the Day 2 bug in miniature


def test_parameters_and_sequential_order():
    net = make_net(np.random.default_rng(0), [4, 5, 3])
    assert len(net.parameters()) == 4                              # W1, b1, W2, b2 (ReLU has none)
    assert [type(l).__name__ for l in net.layers] == ["Linear", "ReLU", "Linear"]
    assert sum(p.data.size for p in net.parameters()) == 4 * 5 + 5 + 5 * 3 + 3


def test_base_module_forward_must_be_overridden():
    with pytest.raises(NotImplementedError):
        layers.Module()(Tensor([1.0]))


# =====================================================================================
# 4. BUG INJECTION: do these tests notice code that is wrong on purpose?
# =====================================================================================

def _buggy_ce(drop_n=False, no_max=False, wrong_sign=False):
    """A copy of softmax_cross_entropy with ONE deliberate mistake."""
    def f(logits, labels):
        z = np.asarray(logits.data)
        N = z.shape[0]
        rows = np.arange(N)
        shifted = z if no_max else z - z.max(axis=1, keepdims=True)    # BUG: no max-subtraction
        exps = np.exp(shifted)
        sums = exps.sum(axis=1, keepdims=True)
        probs = exps / sums
        loss = -np.log(probs[rows, labels]).mean()
        out = Tensor(loss, (logits,), "softmax_ce")

        def _backward():
            onehot = np.zeros_like(probs)
            onehot[rows, labels] = 1.0
            sign = 1.0 if wrong_sign else -1.0
            grad = (probs + sign * onehot) / (1 if drop_n else N)       # BUG: wrong sign / forgot the /N
            logits.grad += grad * out.grad
        out._backward = _backward
        return out
    return f


def _bug_sigmoid(self):
    s = 1.0 / (1.0 + np.exp(-self.data))
    out = Tensor(s, (self,), "sigmoid")

    def _backward():
        self.grad += s * out.grad                       # BUG: forgot the (1 - s) factor
    out._backward = _backward
    return out


def _bug_linear_forward(self, x):
    return x @ self.W + Tensor(self.b.data)             # BUG: the bias is copied, so it never gets a gradient


def _bug_linear_init(self, in_features, out_features, rng=None, gain=1.0):
    rng = np.random.default_rng() if rng is None else rng
    self.in_features, self.out_features = in_features, out_features
    self.W = Tensor(rng.normal(size=(in_features, out_features)) * gain * np.sqrt(1.0 / in_features))   # BUG: 1/n, not 2/n
    self.b = Tensor(np.zeros(out_features))


def _bug_mse(pred, target):
    if not isinstance(target, Tensor):
        target = Tensor(target)
    return ((pred - target) ** 2).sum()                 # BUG: sum instead of mean (still differentiates correctly!)


def _what_notices_the_bug():
    """Run the checks that do not depend on a specific bug. Return a short text of what failed, or '' if nothing."""
    failed = []
    try:                                                # numerical gradient checks
        for name, build in NUMERIC_CASES:
            loss_fn, tensors = build()
            if numerical_check(loss_fn, tensors) >= TOL:
                failed.append("numerical gradient check")
                break
    except Exception:
        failed.append("numerical gradient check (crash)")
    try:                                                # huge-logit stability test
        z = Tensor(np.array([[1000.0, 1001.0, 1002.0]]))
        loss = losses.softmax_cross_entropy(z, np.array([2]))
        loss.backward()
        if not (np.isfinite(loss.data) and np.isfinite(z.grad).all()):
            failed.append("huge-logit stability test")
    except Exception:
        failed.append("huge-logit stability test (crash)")
    try:                                                # He-init spread
        layer = layers.Linear(2000, 500, np.random.default_rng(0))
        if abs(layer.W.data.std() / math.sqrt(2.0 / 2000) - 1.0) >= 0.03:
            failed.append("He-init spread test")
    except Exception:
        failed.append("He-init spread test (crash)")
    try:                                                # gradient sign test (my paper derivation)
        z = Tensor(LOGITS.copy())
        losses.softmax_cross_entropy(z, LABELS).backward()
        if not np.allclose(z.grad.sum(axis=1), 0.0, atol=1e-12) or not (z.grad[np.arange(8), LABELS] < 0).all():
            failed.append("gradient sign / row-sum test")
    except Exception:
        failed.append("gradient sign test (crash)")
    if torch is not None:                               # comparison with PyTorch
        try:
            mine = losses.mse_loss(Tensor(PRED), Tensor(TARGET)).data
            theirs = F.mse_loss(torch.tensor(PRED), torch.tensor(TARGET)).item()
            if scaled_error(mine, theirs) >= TOL:
                failed.append("PyTorch comparison (mse)")
        except Exception:
            failed.append("PyTorch comparison (crash)")
    return ", ".join(failed)


BUGS = [
    ("CE: forgot the /N", "grad = p - onehot (no division by N)", "ce", dict(drop_n=True)),
    ("CE: no max-subtraction", "naive softmax, exp(1000) overflows", "ce", dict(no_max=True)),
    ("CE: wrong sign", "grad = p + onehot instead of p - onehot", "ce", dict(wrong_sign=True)),
    ("Sigmoid: forgot (1 - s)", "derivative s instead of s(1-s)", "sigmoid", None),
    ("Linear: bias detached", "bias never receives a gradient", "linear_forward", None),
    ("Linear: init sqrt(1/n)", "Xavier-style spread used with ReLU", "linear_init", None),
    ("mse: sum instead of mean", "loss is N times too big", "mse", None),
]


@pytest.mark.parametrize("bug", BUGS, ids=[b[0] for b in BUGS])
def test_injected_bug_is_detected(bug, monkeypatch):
    label, description, kind, options = bug
    assert _what_notices_the_bug() == "", "my checks must be clean on the correct code first"
    if kind == "ce":
        monkeypatch.setattr(losses, "softmax_cross_entropy", _buggy_ce(**options))
    elif kind == "sigmoid":
        monkeypatch.setattr(Tensor, "sigmoid", _bug_sigmoid)
    elif kind == "linear_forward":
        monkeypatch.setattr(layers.Linear, "forward", _bug_linear_forward)
    elif kind == "linear_init":
        monkeypatch.setattr(layers.Linear, "__init__", _bug_linear_init)
    elif kind == "mse":
        monkeypatch.setattr(losses, "mse_loss", _bug_mse)
    if kind == "mse" and torch is None:
        pytest.skip("this bug is only visible when comparing with PyTorch")
    with np.errstate(all="ignore"):                     # the no-max bug produces overflow warnings; that is the point
        noticed = _what_notices_the_bug()
    report.record_mutation(label, description, bool(noticed), noticed or "NOTHING")
    assert noticed, f"bug '{label}' slipped through"
