"""Auxiliary functions. Nothing in this file is novel."""

import importlib

import torch
from torch import Tensor
from torch.nn import BatchNorm1d, Linear, Module, Sequential
import random
import numpy as np


class _InputConcatWrapper(Module):
    """Wraps a module and concatenates its input to its output."""

    def __init__(self, module: Module):
        super().__init__()
        self.module = module

    def forward(self, x: Tensor) -> Tensor:
        return torch.cat((x, self.module(x)), dim=-1)


def _make_activation(activation: torch.nn.Module) -> Module:
    if activation in {torch.nn.Softmax, torch.nn.LogSoftmax}:
        return activation(dim=-1)
    return activation()


def cross_cov(x: Tensor, y: Tensor, correction: int = 1, center=True) -> Tensor:
    """Computes the cross-covariance matrix between x and y.

    Args:
        x: Tensor of shape (..., n, d).
        y: Tensor of shape (..., n, d).

    Returns:
        cross_cov: Tensor of shape (..., d, d).
    """
    if center:
        x = x - x.mean(dim=-2, keepdim=True)
        y = y - y.mean(dim=-2, keepdim=True)

    n = x.shape[-2]
    n_corrected = n - correction

    cross_cov = torch.transpose(x, -2, -1) @ y / n_corrected

    return cross_cov


def test_cross_cov() -> None:
    # 2D case (unbatched)
    x = torch.normal(mean=0.0, std=1.0, size=(10, 2))
    assert torch.allclose(cross_cov(x, x), torch.cov(x.T))

    # 3D case (batched)
    batch_size = 5
    x_batch = torch.normal(mean=0.0, std=1.0, size=(batch_size, 50000, 2))
    assert torch.allclose(cross_cov(x_batch, x_batch), torch.stack([torch.cov(x.T) for x in x_batch]))

    # 3D case, shape (1, 2, 2)
    x_batch = torch.arange(4, dtype=float).reshape(2, 2).expand(1, 2, 2)
    y_batch = torch.arange(4, 8, dtype=float).reshape(2, 2).expand(1, 2, 2)
    gt = torch.Tensor([[[12, 14], [22, 26]]]).type_as(x_batch) / 2.0
    assert torch.allclose(cross_cov(x_batch, y_batch, center=False, correction=0), gt)


def whitening_matrix(x: Tensor, regularization: float = 1e-6, sqrt: bool = True) -> Tensor:
    """Compute the whitening matrix from observations of a random vector."""
    assert x.ndim == 2

    n, d = x.shape
    cov_x = cross_cov(x, x)

    # Regularize to improve conditioning
    cov_x_reg = (1 - regularization) * cov_x + regularization * torch.eye(d, device=x.device, dtype=x.dtype)

    L, Q = torch.linalg.eigh(cov_x_reg)

    # Filter eigenvalues
    tol = torch.finfo(L.dtype).eps
    if sqrt:
        sqrt_L_inv = torch.where(L > tol, 1 / L.sqrt(), 0)

        sqrt_cov_x_inv = (Q * sqrt_L_inv) @ Q.T

        return sqrt_cov_x_inv
    else:
        L_inv = torch.where(L > tol, 1 / L, 0)

        cov_x_inv = (Q * L_inv) @ Q.T

        return cov_x_inv


def test_whitening_matrix() -> None:
    x = torch.normal(mean=0.0, std=10.0, size=(10000, 10))
    A = torch.rand(10, 10)
    print(f"cond(A): {torch.linalg.cond(A):.2f}")
    x = x @ A
    cov = cross_cov(x, x, center=False)
    sqrt_cov_inv = whitening_matrix(x, regularization=1e-6)
    Id_approx = sqrt_cov_inv @ cov @ sqrt_cov_inv
    assert torch.allclose(Id_approx, torch.eye(10), atol=1e-3), Id_approx.where(Id_approx.abs() > 1e-3, 0)


def orthonormal_logfro_reg(x: Tensor) -> Tensor:
    r"""Orthonormality regularization with log-Frobenious norm of covariance of x by :footcite:t:`Kostic2023DPNets`.

    .. math::

        \frac{1}{D}\text{Tr}(C_X^{2} - C_X -\ln(C_X)).

    Args:
        x (Tensor): Input features.

    Shape:
        ``x``: :math:`(N, D)`, where :math:`N` is the batch size and :math:`D` is the number of features.
    """
    cov = cross_cov(x, x)  # shape: (D, D)
    eps = torch.finfo(cov.dtype).eps * cov.shape[0]
    vals_x = torch.linalg.eigvalsh(cov)
    vals_x = torch.where(vals_x > eps, vals_x, eps)
    reg = torch.mean(-torch.log(vals_x) + vals_x * (vals_x - 1.0))
    return reg


def make_mlp(
    input_dim: int,
    output_dim: int,
    layer_size: int,
    n_hidden: int,
    activation: torch.nn.Module,
    output_activation: torch.nn.Module = torch.nn.Tanh,
    use_batchnorm: bool = False,
    residual: bool = False,
) -> Module:
    
    # Determine nonlinearity for kaiming initialization based on activation function
    activation_name = activation.__name__.lower() if hasattr(activation, '__name__') else str(activation).lower()
    if 'relu' in activation_name:
        nonlinearity = 'relu'
    elif 'leaky' in activation_name:
        nonlinearity = 'leaky_relu'
    else:
        nonlinearity = 'relu'  # default fallback
    
    init_layer = Linear(input_dim, layer_size)
    torch.nn.init.kaiming_uniform_(init_layer.weight, nonlinearity=nonlinearity)
    torch.nn.init.zeros_(init_layer.bias)

    layers = [init_layer, _make_activation(activation)]

    for _ in range(n_hidden - 1):
        hidden_layer = Linear(layer_size, layer_size)
        torch.nn.init.kaiming_uniform_(hidden_layer.weight, nonlinearity=nonlinearity)
        torch.nn.init.zeros_(hidden_layer.bias)

        layers += [hidden_layer, _make_activation(activation)]

    output_layer = Linear(layer_size, output_dim)
    torch.nn.init.xavier_uniform_(output_layer.weight)
    torch.nn.init.zeros_(output_layer.bias)
    
    layers += [output_layer, _make_activation(output_activation)]

    mlp = Sequential(*layers)

    if residual:
        return _InputConcatWrapper(mlp)

    return mlp


class Symmetric(Module):
    """Torch parametrization to ensure a module is symmetric."""

    def forward(self, X):
        return X.triu() + X.triu(1).transpose(-1, -2)


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")

    def early_stop(self, validation_loss):
        # If the reduction is bigger than min_delta, reset counter
        if self.min_validation_loss - validation_loss > self.min_delta:
            self.min_validation_loss = validation_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def class_from_name(module_name, class_name):
    # load the module, will raise ImportError if module cannot be loaded
    m = importlib.import_module(module_name)
    # get the class, will raise AttributeError if class cannot be found
    c = getattr(m, class_name)
    return c

def set_seed(seed: int=42):
    """
    Set seed for reproducibility in numpy and torch (CPU + CUDA).
    """
    # Python random (optional but recommended)
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch (CPU)
    torch.manual_seed(seed)

    # PyTorch (GPU)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Make CuDNN deterministic (slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False