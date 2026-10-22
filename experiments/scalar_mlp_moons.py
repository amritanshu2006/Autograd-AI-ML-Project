import random

import matplotlib
matplotlib.use("Agg")  # save plots to files; no window needed
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons

from autograd.value import Value
from nn.scalar_mlp import MLP


def mse_loss(model, X, y):
    preds = [model(x) for x in X]
    total = sum(((p - t) ** 2 for p, t in zip(preds, y)), Value(0.0))
    return total * (1.0 / len(X)), preds


def accuracy(preds, y):
    return sum((p.data > 0) == (t > 0) for p, t in zip(preds, y)) / len(y)


def train(model, X, y, lr=0.3, steps=100, use_zero_grad=True, verbose=True):
    history = []
    for k in range(steps):
        loss, preds = mse_loss(model, X, y)      # forward
        if use_zero_grad:
            model.zero_grad()                    # reset old gradients
        loss.backward()                          # backward
        for p in model.parameters():             # update
            p.data -= lr * p.grad
        history.append(loss.data)
        if verbose and (k % 10 == 0 or k == steps - 1):
            print(f"step {k:3d}  loss {loss.data:.4f}  acc {accuracy(preds, y):.2f}")
    return history


def make_data(n=100, seed=1):
    Xa, ya = make_moons(n_samples=n, noise=0.1, random_state=seed)
    return Xa.tolist(), (ya * 2 - 1).tolist()   # labels become -1 / +1


def plot_boundary(model, X, y, path):
    Xa = np.array(X)
    xs = np.linspace(Xa[:, 0].min() - 0.5, Xa[:, 0].max() + 0.5, 40)
    ys = np.linspace(Xa[:, 1].min() - 0.5, Xa[:, 1].max() + 0.5, 40)
    xx, yy = np.meshgrid(xs, ys)
    scores = np.array([[model([a, b]).data for a, b in zip(row_x, row_y)]
                       for row_x, row_y in zip(xx, yy)])
    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, scores > 0, alpha=0.3, cmap="coolwarm")
    plt.scatter(Xa[:, 0], Xa[:, 1], c=y, cmap="coolwarm", edgecolors="k")
    plt.title("Scalar MLP decision boundary (make_moons)")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_loss_curves(hist_ok, hist_bad, path):
    plt.figure(figsize=(6, 4))
    plt.plot(hist_ok, label="with zero_grad")
    plt.plot(hist_bad, label="without zero_grad")
    plt.yscale("log")
    plt.xlabel("step")
    plt.ylabel("loss (log scale)")
    plt.legend()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    random.seed(1337)
    X, y = make_data(n=100)

    model = MLP(2, [8, 8, 1])
    hist_ok = train(model, X, y, lr=0.3, steps=100)
    plot_boundary(model, X, y, "notebooks_or_plots/day2_moons_boundary.png")

    # Same data, same learning rate, but gradients are never reset.
    random.seed(1337)
    bad_model = MLP(2, [8, 8, 1])
    hist_bad = train(bad_model, X, y, lr=0.3, steps=30, use_zero_grad=False, verbose=False)
    plot_loss_curves(hist_ok[:30], hist_bad, "notebooks_or_plots/day2_zero_grad_bug.png")
    print("final loss with zero_grad:", round(hist_ok[-1], 4))
    print("loss after 30 steps without zero_grad:", hist_bad[-1])
