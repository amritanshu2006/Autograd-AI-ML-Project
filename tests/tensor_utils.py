"""Helpers for gradient-checking Tensor operations."""
import time
import numpy as np

from autograd.tensor import Tensor
from tests import report

TOL = 1e-6   # a check passes if scaled error < TOL
EPS = 1e-6   # step size for central differences


def scaled_error(a, n):
    """max|a-n| / max(1, |a|, |n|): relative for big grads, absolute for small."""
    a, n = np.asarray(a, float), np.asarray(n, float)
    denom = max(1.0, np.max(np.abs(a)), np.max(np.abs(n)))
    return float(np.max(np.abs(a - n)) / denom)


def numerical_gradient(f, arrays, i):
    """Central difference of scalar f w.r.t. arrays[i], one entry at a time."""
    x = arrays[i]
    g = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        old = x[idx]
        x[idx] = old + EPS
        plus = f(*arrays)
        x[idx] = old - EPS
        minus = f(*arrays)
        x[idx] = old                      # always restore
        g[idx] = (plus - minus) / (2 * EPS)
    return g


def analyze_gradients(fn, arrays, seed=1234):
    """Compare autograd vs numerical gradients for every input of fn.

    fn takes Tensors, returns a Tensor. The scalar loss is sum(W * fn(...))
    with random W, so each output element gets a different upstream gradient
    (an all-ones gradient can hide bugs).
    """
    arrays = [np.array(a, dtype=float) for a in arrays]
    t0 = time.perf_counter()
    out_shape = fn(*[Tensor(a) for a in arrays]).data.shape
    W = np.random.default_rng(seed).normal(size=out_shape)

    tensors = [Tensor(a) for a in arrays]
    loss = (fn(*tensors) * Tensor(W)).sum()
    loss.backward()

    def scalar(*arrs):
        return float(np.sum(W * fn(*[Tensor(a) for a in arrs]).data))

    stats = []
    for i, t in enumerate(tensors):
        num = numerical_gradient(scalar, [a.copy() for a in arrays], i)
        shape_ok = t.grad.shape == arrays[i].shape
        stats.append({
            "input": i,
            "shape_ok": shape_ok,
            "abs": float(np.max(np.abs(t.grad - num))) if shape_ok else float("inf"),
            "scaled": scaled_error(t.grad, num) if shape_ok else float("inf"),
        })
    return stats, time.perf_counter() - t0


def shapes_text(arrays):
    return " , ".join(str(tuple(np.shape(a))).replace(" ", "") for a in arrays)


def run_check(category, name, fn, arrays):
    stats, secs = analyze_gradients(fn, arrays)
    worst = max(s["scaled"] for s in stats)
    ok = all(s["shape_ok"] for s in stats) and worst < TOL
    report.record_check(category, name, shapes_text(arrays), stats, worst, ok, secs)
    return ok, stats