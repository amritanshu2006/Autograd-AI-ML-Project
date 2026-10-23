# Day 3 - Tensor gradient report

- gradient checks passed: **37/37**
- worst scaled error: **6.8e-10** (about 9 correct digits)
- injected bugs caught: **11/11**

## Per-operation checks

| category | check | shapes | scaled error | status |
|---|---|---|---|---|
| arithmetic | add same shape | (3,4) , (3,4) | 1.3e-10 | PASS |
| arithmetic | mul same shape | (3,4) , (3,4) | 1.9e-10 | PASS |
| arithmetic | sub | (3,4) , (3,4) | 1.5e-10 | PASS |
| arithmetic | div | (3,4) , (3,4) | 9.1e-11 | PASS |
| arithmetic | neg | (3,4) | 7.9e-11 | PASS |
| arithmetic | pow 3 | (3,4) | 1.7e-10 | PASS |
| arithmetic | scalar on the left | (3,4) | 1.3e-10 | PASS |
| arithmetic | reflected div | (3,4) | 8.8e-11 | PASS |
| broadcasting | add (3,4)+(4,) | (3,4) , (4,) | 2.9e-10 | PASS |
| broadcasting | add (3,4)+(3,1) | (3,4) , (3,1) | 1.6e-10 | PASS |
| broadcasting | mul (3,4)*(4,) | (3,4) , (4,) | 1.6e-10 | PASS |
| broadcasting | mul (3,1)*(1,4) | (3,1) , (1,4) | 1.9e-10 | PASS |
| broadcasting | add scalar-shape () | (3,4) , () | 8.4e-11 | PASS |
| broadcasting | mul (2,3,4)*(4,) | (2,3,4) , (4,) | 1.6e-10 | PASS |
| matmul | (2,3)@(3,4) | (2,3) , (3,4) | 1.9e-10 | PASS |
| matmul | (1,5)@(5,1) | (1,5) , (5,1) | 3.3e-11 | PASS |
| matmul | (4,4)@(4,4) | (4,4) , (4,4) | 3.7e-10 | PASS |
| matmul | (6,2)@(2,3) | (6,2) , (2,3) | 6.8e-10 | PASS |
| reductions | sum all | (3,4) | 8.9e-11 | PASS |
| reductions | sum axis=0 | (3,4) | 1.6e-10 | PASS |
| reductions | sum axis=1 | (3,4) | 4.0e-10 | PASS |
| reductions | sum axis=1 keepdims | (3,4) | 1.3e-10 | PASS |
| reductions | mean all | (3,4) | 1.8e-11 | PASS |
| reductions | mean axis=0 | (3,4) | 4.5e-11 | PASS |
| elementwise | relu | (3,4) | 6.3e-11 | PASS |
| elementwise | tanh | (3,4) | 2.5e-10 | PASS |
| elementwise | exp | (3,4) | 6.6e-11 | PASS |
| elementwise | log | (3,4) | 1.6e-10 | PASS |
| composite | mean(relu(X@W+b)) | (5,3) , (3,4) , (4,) | 7.3e-11 | PASS |
| composite | two-layer tanh net | (4,3) , (3,5) , (5,) , (5,2) , (2,) | 2.7e-10 | PASS |
| composite | reused tensor a*a+a | (3,4) | 1.4e-10 | PASS |
| composite | shared weight W used twice | (3,3) , (3,3) | 1.9e-10 | PASS |
| composite | x / x.sum (normalise) | (3,4) | 1.7e-10 | PASS |
| composite | softmax rows | (3,4) | 8.0e-11 | PASS |
| composite | mse loss | (6,1) , (6,1) | 5.0e-10 | PASS |
| composite | log-sum-exp style | (3,4) | 2.3e-10 | PASS |
| composite | bias grad via matmul | (4,2) , (2,3) , (1,3) | 4.7e-10 | PASS |

## Bug injection

| bug | description | caught |
|---|---|---|
| add: no unbroadcast | gradient not summed over broadcast axes | yes |
| mul: no unbroadcast | gradient not summed over broadcast axes | yes |
| mul: = instead of += | reused tensors lose gradient | yes |
| matmul: dA missing .T | dA = dC @ B instead of dC @ B.T | yes |
| matmul: dB wrong order | dB = dC.T @ A instead of A.T @ dC | yes |
| matmul: forward off | forward value shifted by 0.001 | yes |
| sum: ignores axis | backward ignores the reduced axis | yes |
| mean: wrong divisor | divides by total size, not axis size | yes |
| relu: no mask | passes gradient where x<=0 | yes |
| tanh: 1-t not 1-t^2 | wrong derivative formula | yes |
| exp: missing e^x | derivative equals upstream only | yes |
