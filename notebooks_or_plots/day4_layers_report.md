# Day 4 - layers and losses report

- PyTorch matches: **10/10**, worst error 2.9e-16
- numerical gradient checks: **10/10**, worst error 1.3e-09
- injected bugs caught: **7/7**

## Comparison with PyTorch

| component | forward | worst gradient | status |
|---|---|---|---|
| Linear (in=5, out=4) | 1.2e-16 | 1.4e-16 | PASS |
| ReLU | 0 | 0 | PASS |
| Tanh | 1.1e-16 | 2.9e-16 | PASS |
| Sigmoid | 5.6e-17 | 5.6e-17 | PASS |
| Sigmoid (extreme inputs) | 2.4e-38 | 0 | PASS |
| mse_loss | 1.2e-16 | 5.4e-20 | PASS |
| softmax_cross_entropy | 1.3e-16 | 2.7e-20 | PASS |
| softmax_cross_entropy (huge logits) | 0 | 3.1e-266 | PASS |
| 3-layer net (Tanh) + cross-entropy | 2.3e-16 | 1.1e-16 | PASS |
| 3-layer net (ReLU) + cross-entropy | 1.3e-16 | 1.1e-16 | PASS |

## Starting loss of a fresh 10-class network (ln 10 = 2.3026)

| network | mean loss | spread |
|---|---|---|
| He init everywhere (last layer gain 1.0) | 3.1250 | 0.268 |
| He init, small last layer (gain 0.01) | 2.3026 | 0.001 |

## Bug injection

| bug | caught | noticed by |
|---|---|---|
| CE: forgot the /N | yes | numerical gradient check |
| CE: no max-subtraction | yes | huge-logit stability test |
| CE: wrong sign | yes | numerical gradient check, gradient sign / row-sum test |
| Sigmoid: forgot (1 - s) | yes | numerical gradient check |
| Linear: bias detached | yes | numerical gradient check |
| Linear: init sqrt(1/n) | yes | He-init spread test |
| mse: sum instead of mean | yes | PyTorch comparison (mse) |
