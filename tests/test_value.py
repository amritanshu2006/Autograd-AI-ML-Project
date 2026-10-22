#from tests.utils import check_grad


#def test_add_mul():
#    check_grad(lambda a, b: a * b + a, [2.0, -3.0])


#def test_reused_value():
#    check_grad(lambda a: a * a, [3.0])   # gradient should be 2a


#def test_tanh_chain():
#    check_grad(lambda x, w, b: (x * w + b).tanh(), [2.0, -3.0, 6.88])

#def test_sub_div():
#    check_grad(lambda a, b: (a - b) / (b * b), [3.0, 2.0])

#def test_pow():
#    check_grad(lambda a: a ** 3, [1.5])

#def test_exp_log():
#    check_grad(lambda a: (a.exp() + 1.0).log(), [0.7])

#def test_relu():
#    check_grad(lambda a, b: (a * b).relu(), [2.0, 3.0])
#    check_grad(lambda a, b: (a * b).relu(), [2.0, -3.0])

#The mentioned above are the fucntions that I developed for the sake of practice, now for the case of real time check I am using the mentioned below code that imports all the cases from tests/cases.py . 
import pytest

from autograd.value import Value
from tests.cases import CASES, IDS
from tests.utils import check_grad


@pytest.mark.parametrize("name,f,xs", CASES, ids=IDS)
def test_matches_numerical(name, f, xs):
    check_grad(f, xs, eps=1e-5, tol=1e-6)


def test_square_gradient_is_2a():
    a = Value(3.0)
    y = a * a
    y.backward()
    assert a.grad == 6.0

