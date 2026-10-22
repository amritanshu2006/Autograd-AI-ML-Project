from autograd.value import Value


def numerical_grad(f, xs, eps=1e-5):
    """Central-difference gradient of f at the point xs."""
    grads = []
    for i in range(len(xs)):
        hi = list(xs)
        lo = list(xs)
        hi[i] += eps
        lo[i] -= eps
        f_hi = f(*[Value(x) for x in hi]).data
        f_lo = f(*[Value(x) for x in lo]).data
        grads.append((f_hi - f_lo) / (2 * eps))
    return grads


def analytic_grad(f, xs):
    """Gradient from our own backward()."""
    vals = [Value(x) for x in xs]
    out = f(*vals)
    out.backward()
    return [v.grad for v in vals]


def check_grad(f, xs, eps=1e-5, tol=1e-6):
    """Assert that backward() matches the central-difference gradient."""
    analytic = analytic_grad(f, xs)
    numeric = numerical_grad(f, xs, eps)
    for i, (a, n) in enumerate(zip(analytic, numeric)):
        scale = max(1.0, abs(a), abs(n))
        assert abs(a - n) <= tol * scale, (
            f"gradient mismatch for input {i}: backward={a}, numerical={n}"
        )

