"""Day 4: the building blocks of a network, written on top of my Tensor engine.

Every layer is a small object with a forward(x) method. Because forward only uses Tensor
operations (@, +, relu, tanh, sigmoid), autograd builds the graph by itself, so I never
write a backward() for a layer. I only write backward rules for single operations.
"""
import numpy as np

from autograd.tensor import Tensor


class Module:
    """Base class: everything in nn/ inherits from this, so all layers share two helpers."""

    def parameters(self):
        # A layer with no weights (like ReLU) has no parameters, so the default is an empty list.
        return []

    def zero_grad(self):
        # Before every backward pass I must clear the old gradients. Otherwise the new gradients are
        # ADDED on top of the old ones (my backward uses +=), which is the bug I saw on Day 2.
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)      # a fresh zero array with the same shape as the weights

    def __call__(self, x):
        # This lets me write layer(x) instead of layer.forward(x).
        return self.forward(x)

    def forward(self, x):
        # Every real layer overrides this. If I forget to, I get a clear error instead of silence.
        raise NotImplementedError


class Linear(Module):
    """y = x @ W + b, with W of shape (in_features, out_features) and He initialization."""

    def __init__(self, in_features, out_features, rng=None, gain=1.0):
        # rng: a NumPy random generator, so my tests can use a fixed seed and get the same weights every time.
        # gain: an extra multiplier on the weight scale (1.0 = plain He). I use a small gain on the LAST layer
        #       so a fresh network starts near the loss ln(10) (the logits start close to 0).
        rng = np.random.default_rng() if rng is None else rng
        self.in_features = in_features
        self.out_features = out_features
        # He initialization: standard deviation sqrt(2 / fan_in), because ReLU kills half of the signal.
        std = gain * np.sqrt(2.0 / in_features)
        # Shape (in, out) means the batch x of shape (N, in) multiplies it directly: (N,in) @ (in,out) -> (N,out).
        self.W = Tensor(rng.normal(size=(in_features, out_features)) * std)
        # Bias starts at 0 (the random weights already break the symmetry). One bias per output neuron.
        self.b = Tensor(np.zeros(out_features))

    def forward(self, x):
        # x @ W gives (N, out). Adding b of shape (out,) broadcasts it over the N rows;
        # my unbroadcast() in the backward pass sums the bias gradient over the batch.
        return x @ self.W + self.b

    def parameters(self):
        return [self.W, self.b]


class ReLU(Module):
    """max(0, x). No parameters."""

    def forward(self, x):
        return x.relu()


class Tanh(Module):
    """tanh(x), squashes to (-1, 1). No parameters."""

    def forward(self, x):
        return x.tanh()


class Sigmoid(Module):
    """1 / (1 + exp(-x)), squashes to (0, 1). No parameters."""

    def forward(self, x):
        return x.sigmoid()


class Sequential(Module):
    """Runs layers one after another: the output of one layer is the input of the next."""

    def __init__(self, *layers):
        self.layers = list(layers)              # keep the layers in the order I gave them

    def forward(self, x):
        for layer in self.layers:               # go through the layers in order
            x = layer(x)                        # feed the previous output into the next layer
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())   # collect the parameters of every layer into one flat list
        return params
