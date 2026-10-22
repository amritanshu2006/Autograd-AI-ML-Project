# Shared test cases: (name, function, input point).
# Each function works on Value objects and on torch tensors, because both
# use the same operators and method names (.tanh(), .exp(), .log(), .relu()).
CASES = [
    ("square", lambda a: a * a, [3.0]),
    ("reuse_add", lambda a: a + a, [2.5]),
    ("reuse_twice", lambda a, b: (a * b + a) * a, [2.0, -3.0]),
    ("reuse_chain", lambda a, b: (a * b + b) * b, [1.5, 2.0]),
    ("neuron_tanh", lambda x1, x2, w1, w2, b: (x1 * w1 + x2 * w2 + b).tanh(),
     [2.0, 0.0, -3.0, 1.0, 6.8813735870195432]),
    ("sub_div", lambda a, b: (a - b) / (b * b), [3.0, 2.0]),
    ("neg", lambda a, b: -a * b, [1.5, 2.5]),
    ("pow", lambda a: a ** 3, [1.5]),
    ("exp_log", lambda a: (a.exp() + 1.0).log(), [0.7]),
    ("relu_pos", lambda a, b: (a * b).relu(), [2.0, 3.0]),
    ("relu_neg", lambda a, b: (a * b).relu(), [2.0, -3.0]),
    ("reflected_ops", lambda a: 2 * a + 1.0 - (1.0 / a), [1.5]),
]
IDS = [c[0] for c in CASES]

