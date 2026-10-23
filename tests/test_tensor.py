"""Day 3 tests for autograd/tensor.py."""
import numpy as np
import pytest

from autograd.tensor import Tensor, unbroadcast
from tests import report
from tests.tensor_utils import TOL, run_check, scaled_error, analyze_gradients

rng = np.random.default_rng(0)


def rand(*shape):
    return rng.normal(size=shape)


def positive(*shape):
    return rng.uniform(0.5, 2.0, size=shape)   # safe for log, pow, division


# (category, name, function of Tensors, list of numpy inputs)
CASES = [
    ("arithmetic",   "add same shape",        lambda a, b: a + b,           [rand(3, 4), rand(3, 4)]),
    ("arithmetic",   "mul same shape",        lambda a, b: a * b,           [rand(3, 4), rand(3, 4)]),
    ("arithmetic",   "sub",                   lambda a, b: a - b,           [rand(3, 4), rand(3, 4)]),
    ("arithmetic",   "div",                   lambda a, b: a / b,           [rand(3, 4), positive(3, 4)]),
    ("arithmetic",   "neg",                   lambda a: -a,                 [rand(3, 4)]),
    ("arithmetic",   "pow 3",                 lambda a: a ** 3,             [rand(3, 4)]),
    ("arithmetic",   "scalar on the left",    lambda a: 2.0 - 3.0 * a,      [rand(3, 4)]),
    ("arithmetic",   "reflected div",         lambda a: 1.0 / a,            [positive(3, 4)]),
    ("broadcasting", "add (3,4)+(4,)",        lambda a, b: a + b,           [rand(3, 4), rand(4)]),
    ("broadcasting", "add (3,4)+(3,1)",       lambda a, b: a + b,           [rand(3, 4), rand(3, 1)]),
    ("broadcasting", "mul (3,4)*(4,)",        lambda a, b: a * b,           [rand(3, 4), rand(4)]),
    ("broadcasting", "mul (3,1)*(1,4)",       lambda a, b: a * b,           [rand(3, 1), rand(1, 4)]),
    ("broadcasting", "add scalar-shape ()",   lambda a, b: a + b,           [rand(3, 4), rand()]),
    ("broadcasting", "mul (2,3,4)*(4,)",      lambda a, b: a * b,           [rand(2, 3, 4), rand(4)]),
    ("matmul",       "(2,3)@(3,4)",           lambda a, b: a @ b,           [rand(2, 3), rand(3, 4)]),
    ("matmul",       "(1,5)@(5,1)",           lambda a, b: a @ b,           [rand(1, 5), rand(5, 1)]),
    ("matmul",       "(4,4)@(4,4)",           lambda a, b: a @ b,           [rand(4, 4), rand(4, 4)]),
    ("matmul",       "(6,2)@(2,3)",           lambda a, b: a @ b,           [rand(6, 2), rand(2, 3)]),
    ("reductions",   "sum all",               lambda a: a.sum(),            [rand(3, 4)]),
    ("reductions",   "sum axis=0",            lambda a: a.sum(axis=0),      [rand(3, 4)]),
    ("reductions",   "sum axis=1",            lambda a: a.sum(axis=1),      [rand(3, 4)]),
    ("reductions",   "sum axis=1 keepdims",   lambda a: a.sum(axis=1, keepdims=True), [rand(3, 4)]),
    ("reductions",   "mean all",              lambda a: a.mean(),           [rand(3, 4)]),
    ("reductions",   "mean axis=0",           lambda a: a.mean(axis=0),     [rand(3, 4)]),
    ("elementwise",  "relu",                  lambda a: a.relu(),           [rand(3, 4) + 0.05]),
    ("elementwise",  "tanh",                  lambda a: a.tanh(),           [rand(3, 4)]),
    ("elementwise",  "exp",                   lambda a: a.exp(),            [rand(3, 4)]),
    ("elementwise",  "log",                   lambda a: a.log(),            [positive(3, 4)]),
    ("composite",    "mean(relu(X@W+b))",     lambda X, W, b: (X @ W + b).relu().mean(),
                                              [rand(5, 3), rand(3, 4), rand(4)]),
    ("composite",    "two-layer tanh net",    lambda X, W1, b1, W2, b2: ((X @ W1 + b1).tanh() @ W2 + b2),
                                              [rand(4, 3), rand(3, 5), rand(5), rand(5, 2), rand(2)]),
    ("composite",    "reused tensor a*a+a",   lambda a: a * a + a,          [rand(3, 4)]),
    ("composite",    "shared weight W used twice", lambda X, W: (X @ W).tanh() @ W,
                                              [rand(3, 3), rand(3, 3)]),
    ("composite",    "x / x.sum (normalise)", lambda a: a / a.sum(axis=1, keepdims=True), [positive(3, 4)]),
    ("composite",    "softmax rows",          lambda a: a.exp() / a.exp().sum(axis=1, keepdims=True), [rand(3, 4)]),
    ("composite",    "mse loss",              lambda p, y: ((p - y) ** 2).mean(), [rand(6, 1), rand(6, 1)]),
    ("composite",    "log-sum-exp style",     lambda a: a.exp().sum(axis=1).log(), [rand(3, 4)]),
    ("composite",    "bias grad via matmul",  lambda X, W, b: X @ W + b, [rand(4, 2), rand(2, 3), rand(1, 3)]),
]
IDS = [f"{c[0]}:{c[1]}" for c in CASES]


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_gradient_matches_numerical(case):
    category, name, fn, arrays = case
    ok, stats = run_check(category, name, fn, arrays)
    for s in stats:
        assert s["shape_ok"], f"{name}: gradient of input {s['input']} has the wrong shape"
    worst = max(s["scaled"] for s in stats)
    assert ok, f"{name}: scaled error {worst:.2e} >= {TOL:.0e} (per input: {[f'{s['scaled']:.1e}' for s in stats]})"


# (name, Tensor function, NumPy function, inputs) - forward values must match NumPy
FORWARD = [
    ("add",     lambda a, b: a + b,                lambda a, b: a + b,                [rand(3, 4), rand(4)]),
    ("mul",     lambda a, b: a * b,                lambda a, b: a * b,                [rand(3, 1), rand(1, 4)]),
    ("matmul",  lambda a, b: a @ b,                lambda a, b: a @ b,                [rand(2, 3), rand(3, 4)]),
    ("sum",     lambda a: a.sum(axis=1),           lambda a: a.sum(axis=1),           [rand(3, 4)]),
    ("mean",    lambda a: a.mean(axis=0),          lambda a: a.mean(axis=0),          [rand(3, 4)]),
    ("relu",    lambda a: a.relu(),                lambda a: np.maximum(a, 0),        [rand(3, 4)]),
    ("tanh",    lambda a: a.tanh(),                np.tanh,                           [rand(3, 4)]),
    ("exp",     lambda a: a.exp(),                 np.exp,                            [rand(3, 4)]),
    ("log",     lambda a: a.log(),                 np.log,                            [positive(3, 4)]),
    ("pow",     lambda a: a ** 3,                  lambda a: a ** 3,                  [rand(3, 4)]),
    ("div",     lambda a, b: a / b,                lambda a, b: a / b,                [rand(3, 4), positive(3, 4)]),
    ("sub",     lambda a, b: a - b,                lambda a, b: a - b,                [rand(3, 4), rand(4)]),
]


@pytest.mark.parametrize("case", FORWARD, ids=[f[0] for f in FORWARD])
def test_forward_matches_numpy(case):
    name, tfn, nfn, arrays = case
    got = tfn(*[Tensor(a) for a in arrays]).data
    want = nfn(*arrays)
    assert got.shape == want.shape, f"{name}: shape {got.shape} != {want.shape}"
    assert np.allclose(got, want, atol=1e-12), f"{name}: forward values differ from NumPy"


# ---------- exact (hand-computed) tests ----------

def test_unbroadcast_exact():
    assert np.array_equal(unbroadcast(np.ones((2, 3)), (3,)), np.array([2., 2., 2.]))
    assert np.array_equal(unbroadcast(np.ones((2, 3)), (2, 1)), np.array([[3.], [3.]]))
    assert np.array_equal(unbroadcast(np.ones((2, 3)), (1, 3)), np.array([[2., 2., 2.]]))
    assert unbroadcast(np.ones((2, 3)), ()) == 6.0
    assert unbroadcast(np.ones((2, 3)), (2, 3)).shape == (2, 3)


def test_bias_gradient_sums_over_batch():
    X = Tensor(np.ones((5, 2)))
    b = Tensor(np.zeros(2))
    (X + b).sum().backward()
    assert np.array_equal(b.grad, np.array([5., 5.])), "bias grad must be the sum over the batch of 5"


def test_matmul_paper_example():
    A = Tensor([[1., 2.], [3., 4.]])
    B = Tensor([[5., 6.], [7., 8.]])
    C = A @ B
    C.grad = np.ones((2, 2))
    C._backward()
    # dA = ones @ B.T ; dB = A.T @ ones
    assert np.array_equal(A.grad, np.array([[11., 15.], [11., 15.]]))
    assert np.array_equal(B.grad, np.array([[4., 4.], [6., 6.]]))


# ---------- behaviour tests ----------

def test_reused_tensor_gradients_accumulate():
    a = Tensor([2.0, 3.0])
    (a * a + a).sum().backward()          # d/da (a^2 + a) = 2a + 1
    assert np.allclose(a.grad, [5.0, 7.0])


def test_numpy_on_the_left_defers_to_tensor():
    t = Tensor([1.0, 2.0])
    r = np.array([10.0, 20.0]) + t
    assert isinstance(r, Tensor), "NumPy array + Tensor must return a Tensor (needs __array_ufunc__ = None)"
    assert np.allclose(r.data, [11.0, 22.0])


def test_deep_graph_does_not_hit_recursion_limit():
    x = Tensor([1.0])
    y = x
    for _ in range(20000):
        y = y + 1.0
    y.backward()                          # a recursive backward() would crash here
    assert np.allclose(x.grad, [1.0])


def test_tensor_copies_input_and_uses_float64():
    raw = np.array([1, 2, 3])
    t = Tensor(raw)
    raw[0] = 99
    assert t.data[0] == 1.0, "Tensor must copy its input"
    assert t.data.dtype == np.float64


def test_matmul_rejects_non_2d_inputs():
    with pytest.raises(AssertionError):
        Tensor(np.ones(3)) @ Tensor(np.ones((3, 2)))


def test_new_tensor_has_zero_gradient_of_matching_shape():
    t = Tensor(np.ones((2, 3)))
    assert t.grad.shape == (2, 3) and not t.grad.any()


# ---------- bug injection: do these tests actually catch broken code? ----------
def _w(x):
    """Wrap a number/array as a Tensor (kept local so the tests don't depend on Tensor internals)."""
    return x if isinstance(x, Tensor) else Tensor(x)


# Each function below replaces one method of Tensor with a deliberately wrong version.

def _bug_add(self, other):
    other = _w(other)
    out = Tensor(self.data + other.data, (self, other), "+")
    def _backward():
        self.grad += out.grad           # BUG: no unbroadcast
        other.grad += out.grad
    out._backward = _backward
    return out


def _bug_mul(self, other):
    other = _w(other)
    out = Tensor(self.data * other.data, (self, other), "*")
    def _backward():
        self.grad += other.data * out.grad          # BUG: no unbroadcast
        other.grad += self.data * out.grad
    out._backward = _backward
    return out


def _bug_mul_overwrite(self, other):
    other = _w(other)
    out = Tensor(self.data * other.data, (self, other), "*")
    def _backward():
        self.grad = unbroadcast(other.data * out.grad, self.data.shape)   # BUG: = instead of +=
        other.grad = unbroadcast(self.data * out.grad, other.data.shape)
    out._backward = _backward
    return out


def _bug_matmul_no_transpose(self, other):
    other = _w(other)
    out = Tensor(self.data @ other.data, (self, other), "@")
    def _backward():
        self.grad += out.grad @ other.data          # BUG: missing .T
        other.grad += self.data.T @ out.grad
    out._backward = _backward
    return out


def _bug_matmul_swapped(self, other):
    other = _w(other)
    out = Tensor(self.data @ other.data, (self, other), "@")
    def _backward():
        self.grad += out.grad @ other.data.T
        other.grad += out.grad.T @ self.data        # BUG: swapped/transposed wrongly
    out._backward = _backward
    return out


def _bug_matmul_forward(self, other):
    other = _w(other)
    out = Tensor(self.data @ other.data + 0.001, (self, other), "@")   # BUG: forward is off
    def _backward():
        self.grad += out.grad @ other.data.T
        other.grad += self.data.T @ out.grad
    out._backward = _backward
    return out


def _bug_sum(self, axis=None, keepdims=False):
    out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,), "sum")
    def _backward():
        self.grad += np.broadcast_to(out.grad, self.data.shape) if axis is None else out.grad.reshape(-1)[0]   # BUG: ignores axis
    out._backward = _backward
    return out


def _bug_mean(self, axis=None, keepdims=False):
    return self.sum(axis=axis, keepdims=keepdims) * (1.0 / self.data.size)   # BUG: always divides by total size


def _bug_relu(self):
    out = Tensor(np.maximum(self.data, 0.0), (self,), "relu")
    def _backward():
        self.grad += out.grad                         # BUG: no mask
    out._backward = _backward
    return out


def _bug_tanh(self):
    t = np.tanh(self.data)
    out = Tensor(t, (self,), "tanh")
    def _backward():
        self.grad += (1 - t) * out.grad               # BUG: 1-t instead of 1-t^2
    out._backward = _backward
    return out


def _bug_exp(self):
    e = np.exp(self.data)
    out = Tensor(e, (self,), "exp")
    def _backward():
        self.grad += out.grad                         # BUG: forgot the factor e^x
    out._backward = _backward
    return out


BUGS = [
    ("add: no unbroadcast",       "__add__",   _bug_add,                "gradient not summed over broadcast axes"),
    ("mul: no unbroadcast",       "__mul__",   _bug_mul,                "gradient not summed over broadcast axes"),
    ("mul: = instead of +=",      "__mul__",   _bug_mul_overwrite,      "reused tensors lose gradient"),
    ("matmul: dA missing .T",     "matmul",    _bug_matmul_no_transpose, "dA = dC @ B instead of dC @ B.T"),
    ("matmul: dB wrong order",    "matmul",    _bug_matmul_swapped,     "dB = dC.T @ A instead of A.T @ dC"),
    ("matmul: forward off",       "matmul",    _bug_matmul_forward,     "forward value shifted by 0.001"),
    ("sum: ignores axis",         "sum",       _bug_sum,                "backward ignores the reduced axis"),
    ("mean: wrong divisor",       "mean",      _bug_mean,               "divides by total size, not axis size"),
    ("relu: no mask",             "relu",      _bug_relu,               "passes gradient where x<=0"),
    ("tanh: 1-t not 1-t^2",       "tanh",      _bug_tanh,               "wrong derivative formula"),
    ("exp: missing e^x",          "exp",       _bug_exp,                "derivative equals upstream only"),
]


def _worst_error_with_current_code():
    """Run every gradient case and forward case; return worst error (inf if anything crashes)."""
    worst = 0.0
    for category, name, fn, arrays in CASES:
        try:
            stats, _ = analyze_gradients(fn, arrays)
            worst = max(worst, max(s["scaled"] for s in stats))
        except Exception:
            return float("inf")            # crash counts as detected
    for name, tfn, nfn, arrays in FORWARD:
        try:
            got = tfn(*[Tensor(a) for a in arrays]).data
            worst = max(worst, scaled_error(got, nfn(*arrays)))
        except Exception:
            return float("inf")
    return worst


@pytest.mark.parametrize("bug", BUGS, ids=[b[0] for b in BUGS])
def test_injected_bug_is_detected(bug, monkeypatch):
    label, method, replacement, description = bug
    # sanity: the same procedure on the unbroken code must be clean
    assert _worst_error_with_current_code() < TOL
    monkeypatch.setattr(Tensor, method, replacement)
    if method == "matmul":
        monkeypatch.setattr(Tensor, "__matmul__", replacement)
    worst = _worst_error_with_current_code()
    caught = worst >= TOL
    report.record_mutation(label, description, caught, worst)
    assert caught, f"bug '{label}' slipped through - the tests are too weak"