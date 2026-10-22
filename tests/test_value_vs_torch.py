import pytest

from autograd.value import Value
from tests.cases import CASES, IDS

torch = pytest.importorskip("torch")


def torch_result(f, xs):
    ts = [torch.tensor(x, dtype=torch.float64, requires_grad=True) for x in xs]
    out = f(*ts)
    out.backward()
    return out.item(), [t.grad.item() for t in ts]


@pytest.mark.parametrize("name,f,xs", CASES, ids=IDS)
def test_matches_torch(name, f, xs):
    vals = [Value(x) for x in xs]
    out = f(*vals)
    out.backward()
    t_out, t_grads = torch_result(f, xs)
    assert abs(out.data - t_out) <= 1e-9 * max(1.0, abs(t_out))
    for i, (v, tg) in enumerate(zip(vals, t_grads)):
        assert abs(v.grad - tg) <= 1e-9 * max(1.0, abs(tg)), (
            f"{name}: input {i}: ours={v.grad}, torch={tg}"
        )

