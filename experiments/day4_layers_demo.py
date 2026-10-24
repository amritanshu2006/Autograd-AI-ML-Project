"""Day 4 demo: I run my layers and losses, print what I learned, and save 6 pictures.

Run it from the repo root:   python -m experiments.day4_layers_demo
The pictures go to notebooks_or_plots/day4_*.png
"""
import math
import os

import matplotlib
matplotlib.use("Agg")                         # draw into files, no window needed
import matplotlib.pyplot as plt
import numpy as np

from autograd.tensor import Tensor
from nn import layers, losses

OUT = "notebooks_or_plots"
os.makedirs(OUT, exist_ok=True)

# ---- one small colour system so every picture looks the same ----
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2de"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   saved {path}")


def say(text):
    print(text)


# =====================================================================================
# 1. Activation functions and their slopes (computed by MY backward pass)
# =====================================================================================
say("\n[1] Activations: I draw each activation and the slope that my backward() computes.")
x = np.linspace(-5, 5, 401)                                   # 401 points from -5 to 5
analytic = {                                                  # the slope formulas I know from the notes
    "ReLU": lambda v: (v > 0).astype(float),                  # 1 where x > 0, else 0
    "Tanh": lambda v: 1 - np.tanh(v) ** 2,                    # 1 - tanh^2
    "Sigmoid": lambda v: (1 / (1 + np.exp(-v))) * (1 - 1 / (1 + np.exp(-v))),   # s(1-s)
}
fig, axes = plt.subplots(2, 3, figsize=(11, 5.6), sharex=True)
for col, (name, cls) in enumerate([("ReLU", layers.ReLU), ("Tanh", layers.Tanh), ("Sigmoid", layers.Sigmoid)]):
    t = Tensor(x.copy())
    y = cls()(t)                                              # my layer's forward
    y.sum().backward()                                        # sum() gives every output an upstream gradient of 1,
                                                              # so t.grad is the slope at each point
    gap = np.abs(t.grad - analytic[name](x)).max()            # compare with the formula
    say(f"   {name:8s} biggest gap between my backward() slope and the formula: {gap:.1e}")
    axes[0, col].plot(x, y.data, color=BLUE, lw=2)
    axes[0, col].set_title(f"{name}: output")
    axes[1, col].plot(x, t.grad, color=ORANGE, lw=2, label="slope from my backward()")
    axes[1, col].plot(x[::20], analytic[name](x[::20]), "o", color=INK, ms=4, label="formula")
    axes[1, col].set_title(f"{name}: slope")
    axes[1, col].set_xlabel("input x")
axes[1, 0].legend(frameon=False, loc="upper left", fontsize=8)
fig.suptitle("What I see: the dots (formula) sit exactly on my autograd slope. ReLU's slope is 0 or 1, "
             "tanh and sigmoid slopes fade away for big |x|", x=0.01, ha="left", fontsize=10, color=MUTED)
fig.tight_layout(rect=(0, 0, 1, 0.95))
save(fig, "day4_activations.png")

# =====================================================================================
# 2. Why the max-subtraction trick matters
# =====================================================================================
say("\n[2] Stability: I add the same number c to every logit. The true loss must not change.")
base = np.array([[1.0, 2.0, 3.0]])                            # one example with 3 classes
label = np.array([2])
shifts = np.arange(0, 1501, 25)                               # c = 0, 25, 50, ... 1500
stable, naive = [], []
for c in shifts:
    z = base + c
    stable.append(float(losses.softmax_cross_entropy(Tensor(z), label).data))   # my fused, stable loss
    with np.errstate(all="ignore"):                                              # hide the overflow warnings
        e = np.exp(z)                                                            # naive softmax: exp of the raw logits
        naive.append(float(-np.log(e[0, 2] / e.sum())))                          # becomes nan once exp overflows
naive = np.array(naive)
first_bad = shifts[np.isnan(naive)][0]
say(f"   stable loss at c=0: {stable[0]:.6f}   at c=1500: {stable[-1]:.6f}   (same, as it should be)")
say(f"   naive loss turns into nan from c = {first_bad} on, because exp(~710) overflows in float64.")
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.plot(shifts, stable, color=BLUE, lw=2.5, label="stable (subtract the max)")
ok = ~np.isnan(naive)
ax.plot(shifts[ok], naive[ok], "o", mfc="none", mec=ORANGE, mew=1.8, ms=7, label="naive exp / sum")
ax.axvline(first_bad, color=MUTED, ls="--", lw=1)
ax.annotate("naive: nan from here\n(exp overflows)", (first_bad, stable[0]), xytext=(first_bad + 60, stable[0] + 0.25),
            color=INK, arrowprops=dict(arrowstyle="->", color=MUTED))
ax.set_xlabel("c added to every logit")
ax.set_ylabel("cross-entropy loss")
ax.set_ylim(0, 1.2)
ax.set_title("Adding c to all logits never changes the loss, but only the stable version survives")
ax.legend(frameon=False)
save(fig, "day4_softmax_stability.png")

# =====================================================================================
# 3. The gradient (p - y) / N, drawn as three small tables
# =====================================================================================
say("\n[3] Gradient: probabilities p, one-hot truth y, and my gradient (p - y)/N for 4 examples, 5 classes.")
rng = np.random.default_rng(1)
N, C = 4, 5
z = Tensor(rng.normal(size=(N, C)) * 1.5)
y = np.array([0, 3, 1, 4])
loss = losses.softmax_cross_entropy(z, y)
loss.backward()
e = np.exp(z.data - z.data.max(axis=1, keepdims=True))
p = e / e.sum(axis=1, keepdims=True)
onehot = np.eye(C)[y]
say(f"   loss = {float(loss.data):.4f}")
say(f"   every row of the gradient adds up to: {np.round(z.grad.sum(axis=1), 12)}   (it must be 0)")
say(f"   the correct class has the only negative number in its row: {[bool((z.grad[i] < 0).sum() == 1 and z.grad[i, y[i]] < 0) for i in range(N)]}")
fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
for ax, (title, M, cmap, vmin, vmax) in zip(axes, [
        ("p = softmax(logits)", p, "Blues", 0, 1),
        ("y = one-hot truth", onehot, "Blues", 0, 1),
        ("gradient = (p - y) / N", z.grad, "RdBu_r", -abs(z.grad).max(), abs(z.grad).max())]):
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    for i in range(N):
        for j in range(C):
            strong = abs(M[i, j]) > 0.55 * max(abs(vmin), abs(vmax))      # dark cell -> use white text, light cell -> dark text
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=9, color="white" if strong else INK)
    ax.set_title(title)
    ax.set_xlabel("class")
    ax.set_ylabel("example")
    ax.set_xticks(range(C))
    ax.set_yticks(range(N))
    ax.grid(False)
fig.suptitle("Blue = push the score up (negative gradient on the correct class), red = push it down. Each row sums to 0.",
             x=0.01, ha="left", fontsize=10, color=MUTED)
fig.tight_layout(rect=(0, 0, 1, 0.93))
save(fig, "day4_ce_gradient.png")

# =====================================================================================
# 4. Signal size through 10 layers with my own Linear + ReLU layers
# =====================================================================================
say("\n[4] Initialization: I push random data through 10 of MY Linear+ReLU layers and track the signal size (RMS).")
width, depth = 256, 10
n_scale = 1.0 / math.sqrt(2.0 / width)                        # turns a wanted std into a 'gain' for my Linear
settings = [("tiny (std 0.01)", 0.01 * n_scale, BLUE),
            ("Xavier sqrt(1/n)", 1 / math.sqrt(2), ORANGE),
            ("He sqrt(2/n)", 1.0, AQUA),
            ("big (std 1.0)", 1.0 * n_scale, YELLOW)]
fig, ax = plt.subplots(figsize=(8.5, 4.6))
for label_, gain, colour in settings:
    r = np.random.default_rng(0)
    h = Tensor(r.normal(size=(512, width)))
    rms = [1.0]                                               # the input has RMS 1
    for _ in range(depth):
        h = layers.ReLU()(layers.Linear(width, width, r, gain=gain)(h))
        rms.append(float(np.sqrt((h.data ** 2).mean())))
    ax.plot(range(depth + 1), rms, "-o", color=colour, lw=2, ms=4, label=label_)
    ax.annotate(f"{rms[-1]:.2g}", (depth, rms[-1]), xytext=(6, 0), textcoords="offset points", va="center", color=INK, fontsize=9,
                bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none"))
    say(f"   {label_:18s} RMS after layer {depth}: {rms[-1]:.3g}")
ax.axhline(1.0, color=MUTED, ls="--", lw=1)
ax.text(0.2, 2.5, "healthy = 1", color=MUTED, fontsize=9)
ax.set_yscale("log")
ax.set_xlim(0, depth + 0.8)
ax.set_xlabel("layer number")
ax.set_ylabel("signal size (RMS), log scale")
ax.set_title("Only He init keeps the signal near 1 through ReLU layers")
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4)
save(fig, "day4_init_signal.png")

# =====================================================================================
# 5. Starting loss of a fresh 10-class network over many seeds
# =====================================================================================
say("\n[5] Starting loss: 100 fresh 10-class networks on random data. The checklist wants about ln(10) = 2.30.")
plain, small = [], []
for seed in range(100):
    r = np.random.default_rng(seed)
    data = Tensor(r.normal(size=(128, 20)))
    labels = r.integers(0, 10, size=128)
    for gain, store in ((1.0, plain), (0.01, small)):
        sizes = [20, 64, 64, 10]
        net = layers.Sequential(layers.Linear(20, 64, np.random.default_rng(seed + 1000)), layers.ReLU(),
                                layers.Linear(64, 64, np.random.default_rng(seed + 2000)), layers.ReLU(),
                                layers.Linear(64, 10, np.random.default_rng(seed + 3000), gain=gain))
        store.append(float(losses.softmax_cross_entropy(net(data), labels).data))
say(f"   plain He on the last layer : mean {np.mean(plain):.3f}  (spread {np.std(plain):.3f})")
say(f"   small last layer (gain .01): mean {np.mean(small):.3f}  (spread {np.std(small):.3f})")
fig, ax = plt.subplots(figsize=(8, 4.2))
bins = np.linspace(2.2, 4.2, 41)
ax.hist(plain, bins=bins, color=ORANGE, alpha=0.9, label="plain He on the last layer", edgecolor=SURFACE, linewidth=1)
ax.hist(small, bins=bins, color=BLUE, alpha=0.9, label="small last layer (gain 0.01)", edgecolor=SURFACE, linewidth=1)
ax.axvline(math.log(10), color=INK, ls="--", lw=1.2)
ax.text(math.log(10) + 0.12, ax.get_ylim()[1] * 0.6, "ln(10) = 2.303", color=INK)
ax.set_xlabel("loss of an untrained network")
ax.set_ylabel("number of networks")
ax.set_title("A small last layer starts every class at about 10%, so the loss starts at ln(10)")
ax.legend(frameon=False, loc="upper right")
save(fig, "day4_init_loss.png")

# =====================================================================================
# 6. How close am I to PyTorch?
# =====================================================================================
try:
    import torch
    import torch.nn.functional as F
    from tests import layer_report, test_layers as T
except ImportError:
    torch = None
if torch is None:
    say("\n[6] PyTorch is not installed here, so I skip the comparison picture.")
else:
    say("\n[6] PyTorch: I run the same checks as tests/test_layers.py and plot the error of every component.")
    layer_report.TORCH.clear()
    T.test_linear_matches_torch()
    for nm, mine, theirs in [("ReLU", lambda a: layers.ReLU()(a), torch.relu), ("Tanh", lambda a: layers.Tanh()(a), torch.tanh),
                             ("Sigmoid", lambda a: layers.Sigmoid()(a), torch.sigmoid)]:
        T.compare_with_torch(nm, mine, theirs, [T.SHIFT], ["x"])
    T.test_mse_loss_matches_torch()
    T.test_softmax_cross_entropy_matches_torch()
    T.test_softmax_cross_entropy_matches_torch_with_huge_logits()
    T.test_whole_network_matches_torch_model()
    T.test_relu_network_matches_torch_model()
    rows = layer_report.TORCH
    names = [r["component"] for r in rows][::-1]
    errs = [max(r["worst"], 1e-17) for r in rows][::-1]       # a log axis cannot show 0, so I floor at 1e-17
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.barh(names, errs, color=BLUE, height=0.6)
    ax.axvline(1e-6, color=ORANGE, lw=2)
    ax.text(0.7e-6, len(names) - 0.6, "checklist limit 1e-6", color=INK, va="center", ha="right")
    ax.set_xscale("log")
    ax.set_xlim(1e-18, 1e-4)
    ax.set_xlabel("worst error vs PyTorch (forward and gradients), log scale; bars at 1e-17 mean exactly 0")
    ax.set_title("Every component is about ten orders of magnitude inside the limit")
    ax.grid(axis="y", visible=False)
    for i, (n_, e_) in enumerate(zip(names, errs)):
        ax.text(e_ * 1.6, i, f"{e_:.0e}", va="center", fontsize=8, color=MUTED)
    save(fig, "day4_error_vs_torch.png")
    say(f"   worst error over all components: {max(r['worst'] for r in rows):.1e}")

say("\nDone. What I should take away:")
say("  - my backward() slopes equal the textbook slopes (picture 1)")
say("  - subtracting the max is what keeps softmax cross-entropy finite (picture 2)")
say("  - the gradient (p - y)/N pushes the correct class up and the others down (picture 3)")
say("  - He init keeps the signal alive, the others do not (picture 4)")
say("  - a fresh network starts at ln(10) when the last layer is small (picture 5)")
say("  - my layers and losses match PyTorch to about 1e-16 (picture 6)")
