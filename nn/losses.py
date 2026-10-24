"""Day 4: loss functions. A loss turns the network output into ONE number that I call backward() on."""
import numpy as np

from autograd.tensor import Tensor


def mse_loss(pred, target):
    """Mean squared error: mean of (pred - target)^2 over every element."""
    # If the target is a plain NumPy array, wrap it in a Tensor (it just doesn't need a gradient).
    if not isinstance(target, Tensor):
        target = Tensor(target)
    diff = pred - target            # element-wise difference (autograd records this operation)
    return (diff ** 2).mean()       # square it, then average over all elements -> a single number


def softmax_cross_entropy(logits, labels):
    """Softmax + cross-entropy as ONE fused operation.

    logits: Tensor of shape (N, C), the raw scores.
    labels: integer array of shape (N,), the correct class of each example.
    Returns a Tensor holding one number: the average of -log(p_correct).
    """
    z = np.asarray(logits.data)
    labels = np.asarray(labels)
    assert z.ndim == 2, "logits must have shape (N, C)"
    assert labels.shape == (z.shape[0],), "labels must have shape (N,)"
    assert labels.min() >= 0 and labels.max() < z.shape[1], "label out of range"
    N = z.shape[0]
    rows = np.arange(N)                                   # [0, 1, ..., N-1], used to pick one value per row

    # --- forward pass (numerically stable) ---
    shifted = z - z.max(axis=1, keepdims=True)            # subtract each row's max so the biggest value is 0 (no overflow)
    exps = np.exp(shifted)                                # now every number is in (0, 1], so exp is safe
    sums = exps.sum(axis=1, keepdims=True)                # the shared denominator, one per row
    probs = exps / sums                                   # softmax probabilities, each row adds up to 1
    log_probs = shifted - np.log(sums)                    # log(softmax) computed directly, so it never becomes log(0)
    loss = -log_probs[rows, labels].mean()                # -log(p_correct), averaged over the batch

    out = Tensor(loss, (logits,), "softmax_ce")

    # --- backward pass: my derivation from the notes, grad = (softmax - onehot) / N ---
    def _backward():
        onehot = np.zeros_like(probs)                     # a matrix of zeros with the shape of the probabilities
        onehot[rows, labels] = 1.0                        # put a 1 at the correct class of each row
        grad = (probs - onehot) / N                       # the result of my paper derivation
        logits.grad += grad * out.grad                    # out.grad is 1.0 for the final loss; += because of accumulation
    out._backward = _backward
    return out
