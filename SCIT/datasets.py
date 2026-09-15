"""Datasets for Independence and Conditional Independence Testing."""

import math
import random

import numpy as np
import pyro.distributions as D
import torch
from scipy import stats
from torch import Tensor
from torch.utils.data import TensorDataset, random_split

import SCIT.distributions as D2


# ======================================================================================================================
# CONDITIONAL INDEPENDENCE TESTING
# ======================================================================================================================
def same(x):
    return x


def cube(x):
    return np.power(x, 3)


def negexp(x):
    return np.exp(-np.abs(x))


def dataset_ci(
    dataset: str,
    sample_size: int,
    conditional_independent: bool,
    device: int,
    stack_xz: bool,
    stack_yz: bool,
    as_ds: bool,
    lengths: list = None,
    seed=None,
    **kwargs,
):
    if dataset == "multidim_linear_ci":
        dz = kwargs["dim_z"]
        x_np, y_np, z_np = multidim_linear_ci(sample_size, conditional_independent, dim_z=dz, seed=seed)
        x = torch.from_numpy(x_np).to(torch.float32).to(device)
        y = torch.from_numpy(y_np).to(torch.float32).to(device)
        z = torch.from_numpy(z_np).to(torch.float32).to(device)
    elif dataset == "he2025_ci":
        dz = kwargs["dim_z"]
        x_np, y_np, z_np = he2025_ci(sample_size, conditional_independent, device=device, seed=seed, dim_z=dz)
        x = torch.from_numpy(x_np).to(torch.float32).to(device)
        y = torch.from_numpy(y_np).to(torch.float32).to(device)
        z = torch.from_numpy(z_np).to(torch.float32).to(device)
    elif dataset == "benchmark_rebuttal_ci":
        dz = kwargs["dim_z"]
        x_np, y_np, z_np = benchmark_rebuttal_ci(sample_size, conditional_independent, dim_z=dz, seed=seed)
        x = torch.from_numpy(x_np).to(torch.float32).to(device)
        y = torch.from_numpy(y_np).to(torch.float32).to(device)
        z = torch.from_numpy(z_np).to(torch.float32).to(device)
    elif dataset == "post_nonlinear_yang2025_ci":
        dz = kwargs["dim_z"]
        x_np, y_np, z_np = post_nonlinear_yang2025_ci_fixed(sample_size, conditional_independent, dim_z=dz, seed=seed)
        x = torch.from_numpy(x_np).to(torch.float32).to(device)
        y = torch.from_numpy(y_np).to(torch.float32).to(device)
        z = torch.from_numpy(z_np).to(torch.float32).to(device)
    elif dataset == "post_nonlinear_cauchy_ci":
        dz = kwargs["dim_z"]
        x_np, y_np, z_np = post_nonlinear_cauchy_ci(sample_size, conditional_independent, dim_z=dz, seed=seed)
        x = torch.from_numpy(x_np).to(torch.float32).to(device)
        y = torch.from_numpy(y_np).to(torch.float32).to(device)
        z = torch.from_numpy(z_np).to(torch.float32).to(device)

    elif dataset == "postnonlinear_nonsmooth_test_ci":
        dz = kwargs["dim_z"]
        x_np, y_np, z_np = postnonlinear_nonsmooth_test_ci(sample_size, conditional_independent, dim_z=dz, seed=seed)
        x = torch.from_numpy(x_np).to(torch.float32).to(device)
        y = torch.from_numpy(y_np).to(torch.float32).to(device)
        z = torch.from_numpy(z_np).to(torch.float32).to(device)

    elif dataset == "post_nonlinear_ren_2025_ci":
        dz = kwargs["dim_z"]
        x_np, y_np, z_np = post_nonlinear_ren_2025_ci(sample_size, conditional_independent, dim_z=dz, seed=seed)
        x = torch.from_numpy(x_np).to(torch.float32).to(device)
        y = torch.from_numpy(y_np).to(torch.float32).to(device)
        z = torch.from_numpy(z_np).to(torch.float32).to(device)

    elif dataset == "post_nonlinear_zhang_ci":
        x, y, z = post_nonlinear_zhang_ci(sample_size, conditional_independent, device, **kwargs, seed=seed)
    elif dataset == "post_nonlinear_dgcit_CI":
        dz = kwargs["dim_z"]
        x_np, y_np, z_np = postnonlinear_dgcit(size=sample_size, conditional_independent=conditional_independent, dz=dz)
        x = torch.from_numpy(x_np).to(torch.float32).to(device)
        y = torch.from_numpy(y_np).to(torch.float32).to(device)
        z = torch.from_numpy(z_np).to(torch.float32).to(device)
    elif dataset == "linear_cauchy_ci":
        dz = kwargs["dim_z"]
        x_np, y_np, z_np = linear_cauchy_ci(sample_size, conditional_independent, dim_z=dz)
        x = torch.from_numpy(x_np).to(torch.float32).to(device)
        y = torch.from_numpy(y_np).to(torch.float32).to(device)
        z = torch.from_numpy(z_np).to(torch.float32).to(device)
    elif dataset == "nonsmooth_conditionals_ci":
        dz = kwargs.get("dim_z", 5)
        noise_std = kwargs.get("noise_std", 0.05)
        x, y, z = nonsmooth_conditionals_ci(sample_size, dim_z=dz, noise_std=noise_std)
        x = x.to(torch.float32).to(device)
        y = y.to(torch.float32).to(device)
        z = z.to(torch.float32).to(device)
    else:
        raise NotImplementedError(f"The dataset [{dataset}] is not implemented!")

    if stack_xz:
        x = torch.hstack([x, z])

    if stack_yz:
        y = torch.hstack([y, z])

    if as_ds:
        return split_dataset(x, y, z, lengths=lengths)
    else:
        return x, y, z


def jointly_guassian_ci(
    sample_size: int,
    conditional_independent: bool,
    device: int,
):
    raise NotImplementedError("Not Implemented Yet!")


def linear_ci(
    sample_size: int,
    conditional_independent: bool,
    dim_z: int = 4,
    seed: int = None,
    normalize: bool = True,
    **kwargs,
):
    """Linear CI dataset."""
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    A = np.array(
        [
            [3, 2, -1, 4],
            [1, 0.2, 0.7, 1],
            [-3, -3, 0.5, 2],
            [4, 2, -3, 1],
        ]
    )
    B = np.array(
        [
            [3, 2.5, -2, 4],
            [1, 1.2, 0.7, 1],
            [-3, 7, 0.3, 2],
            [-4, -2, -2, 1],
        ]
    )

    z = np.random.normal(0.0, 1.0, size=(sample_size, dim_z))
    noise_x = np.random.normal(0.0, 1.0, size=(sample_size, dim_z))
    noise_y = np.random.normal(0.0, 1.0, size=(sample_size, dim_z))

    x = z @ A[:dim_z, :dim_z].T + noise_x
    y = z @ B[:dim_z, :dim_z].T + noise_y

    if not conditional_independent:
        noise_shared = 5 * np.random.normal(0.0, 1.0, size=(sample_size, dim_z))
        x += noise_shared
        y += noise_shared

    if normalize:
        x = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-8)
        y = (y - y.mean(axis=0)) / (y.std(axis=0) + 1e-8)
        z = (z - z.mean(axis=0)) / (z.std(axis=0) + 1e-8)

    return x, y, z


def postnonlinear_nonsmooth_test_ci(
    sample_size: int,
    conditional_independent: bool,
    dim_z: int,
    seed: int = None,
    str_con_dep: float = 1.5,
    nstd_indep_noise: float = 0.005,
    normalize: bool = True,
    **kwargs,
):
    """Post-nonlinear with non-smooth transformation CI dataset."""
    def function_f(x):
        activation_function = np.where(np.abs(x) < 1, np.cos(2 * np.pi / x), x**2)
        correction_for_nan = np.nan_to_num(activation_function, nan=1)
        return correction_for_nan

    def function_g(x):
        activation_function = np.where(np.abs(x) < 1, np.cos(2 * np.pi / x), x**3)
        correction_for_nan = np.nan_to_num(activation_function, nan=1)
        return correction_for_nan

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    z = np.random.normal(loc=0.0, scale=1.0, size=(sample_size, dim_z))
    z_m = np.mean(z, axis=1).reshape(-1, 1)

    noise_x = np.random.normal(loc=0.0, scale=1.0, size=(sample_size, 1))
    noise_y = np.random.normal(loc=0.0, scale=1.0, size=(sample_size, 1))

    x = z_m + nstd_indep_noise * noise_x
    y = z_m + nstd_indep_noise * noise_y

    if not conditional_independent:
        eb = str_con_dep * np.random.normal(loc=0.0, scale=1.0, size=(sample_size, 1))
        x += eb
        y += eb

    x, y = function_f(x), function_g(y)

    if normalize:
        x = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-8)
        y = (y - y.mean(axis=0)) / (y.std(axis=0) + 1e-8)
        z = (z - z.mean(axis=0)) / (z.std(axis=0) + 1e-8)

    return x, y, z


def multidim_linear_ci(
    sample_size: int,
    conditional_independent: bool,
    dim_z: int,
    seed: int = None,
    str_con_dep: float = 4.0,
    str_Z: float = 0.5,
    noise_str: float = 1.0,
    normalize: bool = True,
    **kwargs,
):
    """Multidimensional linear CI dataset with nonlinear transformations."""
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    z = np.random.normal(loc=0.0, scale=1.0, size=(sample_size, dim_z)) * str_Z

    noise_x = np.random.normal(size=(sample_size, dim_z)) * noise_str
    noise_y = np.random.normal(size=(sample_size, dim_z)) * noise_str

    x = z + noise_x
    y = z + noise_y

    def function_f(x):
        activation_function = np.where(np.abs(x) < 1, np.cos(2 * np.pi / x), x**2)
        correction_for_nan = np.nan_to_num(activation_function, nan=1)
        return correction_for_nan

    def function_g(x):
        activation_function = np.where(np.abs(x) < 1, np.cos(2 * np.pi / x), x**3)
        correction_for_nan = np.nan_to_num(activation_function, nan=1)
        return correction_for_nan

    x, y = function_f(x), function_g(y)

    if not conditional_independent:
        eb = np.random.normal(loc=0.0, scale=str_con_dep, size=(sample_size, 1))
        x += eb
        y += eb

    if normalize:
        x = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-8)
        y = (y - y.mean(axis=0)) / (y.std(axis=0) + 1e-8)
        z = (z - z.mean(axis=0)) / (z.std(axis=0) + 1e-8)

    return x, y, z


def benchmark_rebuttal_ci(
    sample_size: int,
    conditional_independent: bool,
    dim_z: int,
    seed: int = None,
    str_con_dep: float = 0.15,
    str_Z: float = 0.1,
    noise_str: float = 0.25,
    normalize: bool = True,
    **kwargs,
):
    """Benchmark rebuttal CI dataset with sin/cos transformations."""
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    z = np.random.normal(loc=0.0, scale=1.0, size=(sample_size, dim_z)) * str_Z

    noise_x = np.random.normal(size=(sample_size, dim_z)) * noise_str
    noise_y = np.random.normal(size=(sample_size, dim_z)) * noise_str

    x = z + noise_x
    y = z + noise_y

    x = np.sin(x)
    y = np.cos(y)

    if not conditional_independent:
        eb = np.random.normal(loc=0.0, scale=str_con_dep, size=(sample_size, x.shape[1]))
        x += eb
        y += eb

    if normalize:
        x = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-8)
        y = (y - y.mean(axis=0)) / (y.std(axis=0) + 1e-8)
        z = (z - z.mean(axis=0)) / (z.std(axis=0) + 1e-8)

    return x, y, z


def post_nonlinear_ren_2025_ci(
    sample_size: int,
    conditional_independent: bool,
    dim_z: int,
    seed: int = None,
    str_con_dep: float = 1.3,
    str_Z: float = 0.0,
    noise_str: float = 1.0,
    f1: str = "cos",
    f2: str = "cos",
    normalize: bool = True,
    **kwargs,
):
    """Dataset for conditional dependence testing (Ren 2025)."""
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    z = np.random.normal(loc=0.0, scale=1.0, size=(sample_size, dim_z))
    z_m = np.mean(z, axis=1).reshape(-1, 1) * str_Z

    noise_x = np.random.normal(loc=0.0, scale=noise_str, size=(sample_size, 1))
    noise_y = np.random.normal(loc=0.0, scale=noise_str, size=(sample_size, 1))

    x = z_m + noise_x
    y = z_m + noise_y

    if not conditional_independent:
        eb = str_con_dep * np.random.normal(loc=0.0, scale=1.0, size=(sample_size, 1))
        x += eb
        y += eb

    funcs = {
        "linear": lambda x: x,
        "square": lambda x: x**2,
        "cos": lambda x: np.cos(x),
        "cube": lambda x: x**3,
        "tanh": lambda x: np.tanh(x),
        "sin": lambda x: np.sin(x),
        "exp": lambda x: np.exp(x),
    }

    func1 = funcs[f1]
    func2 = funcs[f2]

    x, y = func1(x), func2(y)

    if normalize:
        x = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-8)
        y = (y - y.mean(axis=0)) / (y.std(axis=0) + 1e-8)
        z = (z - z.mean(axis=0)) / (z.std(axis=0) + 1e-8)

    return x, y, z


def nonsmooth_conditionals_ci(
    sample_size: int,
    conditional_independent: bool,
    dim_z: int,
    seed: int = None,
    noise_std: float = 0.05,
    normalize: bool = True,
    **kwargs,
) -> tuple:
    # Sample Z from standard Gaussian
    z = np.random.randn(sample_size, dim_z)

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    # Determine the quadrant of Z
    signs = np.sign(z)

    # Map each quadrant to a different mean for X and Y (fixed per quadrant)
    proj_rng = np.random.default_rng(2026)
    proj_x = proj_rng.standard_normal((dim_z, 1)) * 5.0
    proj_y = proj_rng.standard_normal((dim_z, 1)) * 5.0

    # We apply a non-linear transformation to the projection to ensure
    # that the means vary "wildly" and are not just linear combinations of signs.
    mu_x = np.exp((signs @ proj_x) * 10.0).squeeze()
    mu_y = np.cos((signs @ proj_y) * 10.0).squeeze()

    # 4. Sample X and Y conditionally independent given Z (via the quadrant)
    x = mu_x + np.random.randn(sample_size) * noise_std
    y = mu_y + np.random.randn(sample_size) * noise_std

    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)

    if normalize:
        x = (x - x.mean(axis=0)) / x.std(axis=0)
        y = (y - y.mean(axis=0)) / y.std(axis=0)
        z = (z - z.mean(axis=0)) / z.std(axis=0)

    return x, y, z


def linear_cauchy_ci(
    sample_size: int,
    conditional_independent: bool,
    dim_z: int,
    seed: int = None,
    normalize: bool = True,
    **kwargs,
) -> tuple:
    """Linear CI dataset with Cauchy distributed Z."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    cauchy = torch.distributions.Cauchy(loc=0.0, scale=1.0)
    z = cauchy.sample((sample_size, dim_z))
    z_mean = z.mean(dim=1, keepdim=True)
    noise_1 = torch.normal(mean=0.0, std=1.0, size=(sample_size, 1))
    noise_2 = torch.normal(mean=0.0, std=1.0, size=(sample_size, 1))

    x = z_mean + noise_1

    if conditional_independent:
        y = z_mean + noise_2
    else:
        y = z_mean + noise_1 + noise_2

    x = x.numpy()
    y = y.numpy()
    z = z.numpy()
    
    if normalize:
        x = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-8)
        y = (y - y.mean(axis=0)) / (y.std(axis=0) + 1e-8)
        z = (z - z.mean(axis=0)) / (z.std(axis=0) + 1e-8)
    
    return x, y, z


def he2025_ci(
    sample_size: int,
    conditional_independent: bool,
    dim_z: int = 3,
    seed: int = None,
    ca_dim_idx: int = 0,
    cb_dim_idx: int = 1,
    cr_dim_idx: int = 2,
    alpha: float = 0.1,
    beta: float = 2.0,
    normalize: bool = True,
    **kwargs,
):
    """Generate data for the conditional independence test (He 2025)."""
    n_points = sample_size
    dim = dim_z
    ground_truth = "H0" if conditional_independent else "H1"
    if seed is None:
        seed = np.random.randint(0, 10000)
    c = np.random.RandomState(seed=seed * (n_points + 1)).normal(0, 1, size=(n_points, dim))

    f = np.cos
    g = np.exp

    a_m = f(c[:, ca_dim_idx : ca_dim_idx + 1])
    b_m = g(c[:, cb_dim_idx : cb_dim_idx + 1])

    if ground_truth == "H1":
        r = np.sin(beta * c[:, cr_dim_idx])
        a_r = np.zeros((n_points, 1))
        b_r = np.zeros((n_points, 1))
        for i in range(n_points):
            cov_matrix = [[1, r[i]], [r[i], 1]]
            a_r[i, 0], b_r[i, 0] = np.random.RandomState(seed=seed * (n_points + 1) + 1 + i).multivariate_normal(
                [0, 0], cov_matrix
            )

    elif ground_truth == "H0":
        a_r = np.random.RandomState(seed=seed * (n_points + 1) + 1).normal(0, 1, size=(n_points, 1))
        b_r = np.random.RandomState(seed=seed * (n_points + 1) + 2).normal(0, 1, size=(n_points, 1))
    else:
        raise NotImplementedError(f"{ground_truth} has to be H0 or H1")

    a = a_m + alpha * a_r
    b = b_m + alpha * b_r

    if normalize:
        a = (a - a.mean(axis=0)) / (a.std(axis=0) + 1e-8)
        b = (b - b.mean(axis=0)) / (b.std(axis=0) + 1e-8)
        c = (c - c.mean(axis=0)) / (c.std(axis=0) + 1e-8)

    return a, b, c


def postnonlinear_dgcit(
    sample_size: int,
    conditional_independent: bool,
    dim_z: int,
    seed: int = None,
    dx: int = 1,
    dy: int = 1,
    nstd: float = 0.5,
    alpha_x: float = 1.4,
    normalize: bool = True,
    dist_z: str = "gaussian",
    **kwargs,
):
    """Generate CI post-nonlinear samples (DGCIT style).
    Z is Gaussian or Laplace, X and Y are sin/cos transformations.
    """
    if seed is None:
        seed = np.random.randint(0, 10000)
    np.random.seed(seed)

    num = sample_size
    dz = dim_z

    if dist_z == "gaussian":
        cov = np.eye(dz)
        mu = np.zeros(dz)
        Z = np.random.multivariate_normal(mu, cov, num)
    elif dist_z == "laplace":
        Z = np.random.laplace(loc=0.0, scale=1.0, size=num * dz)
        Z = np.reshape(Z, (num, dz))
    else:
        raise ValueError(f"Unknown dist_z: {dist_z}")

    Ax = np.random.rand(dz, dx)
    for i in range(dx):
        Ax[:, i] = Ax[:, i] / (np.linalg.norm(Ax[:, i], ord=1) + 1e-8)
    Ay = np.random.rand(dz, dy)
    for i in range(dy):
        Ay[:, i] = Ay[:, i] / (np.linalg.norm(Ay[:, i], ord=1) + 1e-8)

    Axy = np.ones((dx, dy)) * alpha_x

    if conditional_independent:
        X = np.sin(np.matmul(Z, Ax) + nstd * np.random.multivariate_normal(np.zeros(dx), np.eye(dx), num))
        Y = np.cos(np.matmul(Z, Ay) + nstd * np.random.multivariate_normal(np.zeros(dy), np.eye(dy), num))
    else:
        X = np.sin(np.matmul(Z, Ax) + nstd * np.random.multivariate_normal(np.zeros(dx), np.eye(dx), num))
        Y = np.cos(
            np.matmul(X, Axy) + np.matmul(Z, Ay) + nstd * np.random.multivariate_normal(np.zeros(dx), np.eye(dx), num)
        )

    if normalize:
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        Y = (Y - Y.mean(axis=0)) / (Y.std(axis=0) + 1e-8)
        Z = (Z - Z.mean(axis=0)) / (Z.std(axis=0) + 1e-8)

    return np.array(X), np.array(Y), np.array(Z)


def post_nonlinear_zhang_ci(
    sample_size: int,
    conditional_independent: bool,
    dim_z: int = 1,
    seed: int = None,
    case: int = 1,
    normalize: bool = True,
    **kwargs,
):
    """Sample from post-nonlinear data model of https://arxiv.org/abs/1202.3775."""

    def F_X(Z):
        return 0.7 * ((Z**3) / 5 + Z / 2)

    def G_X(W):
        return W + (W**3) / 3 + np.tanh(W / 3) / 2

    def F_Y(Z):
        return ((Z**3) / 4 + Z) / 3

    def G_Y(W):
        return W + np.tanh(W / 3)

    def H(W):
        return W / 2 + 0.7 * np.tanh(W)

    def I(W, Q):
        return W * 2 / 3 + Q * 5 / 6

    if seed is None:
        seed = np.random.randint(0, 10000)
    np.random.seed(seed)
    random.seed(seed)

    dim = dim_z
    noise_x = np.random.normal(loc=0.0, scale=1.0, size=(sample_size, 1))
    noise_y = np.random.normal(loc=0.0, scale=1.0, size=(sample_size, 1))
    z = np.random.normal(loc=0.0, scale=1.0, size=(sample_size, dim))

    zz1 = F_X(z[:, 0].reshape(-1, 1))
    zz2 = F_Y(z[:, 0].reshape(-1, 1))

    if dim > 1 and case == 2:
        zz1 = H(zz1 / 2 + z[:, 1].reshape(-1, 1))
        zz2 = H(zz2 / 2 + z[:, 1].reshape(-1, 1))

        for i in range(2, dim):
            z_i = z[:, i].reshape(-1, 1)
            zz1 = H(I(zz1, z_i))
            zz2 = H(I(zz2, z_i))

    x = G_X(zz1 + np.tanh(noise_x))
    y = G_Y(zz2 + noise_y)

    # Add shared noise to make X and Y conditionally dependent
    if not conditional_independent:
        ff = np.random.normal(loc=0.0, scale=1.0, size=(sample_size, 1)) * 0.5
        x = x + ff
        y = y + ff

    if normalize:
        x = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-8)
        y = (y - y.mean(axis=0)) / (y.std(axis=0) + 1e-8)
        z = (z - z.mean(axis=0)) / (z.std(axis=0) + 1e-8)

    return x, y, z


def post_nonlinear_cauchy_ci(
    sample_size: int,
    conditional_independent: bool,
    dim_z: int,
    seed: int = None,
    noise_scale: float = 0.5,
    cond_dep_noise_scale: float = 1.0,
    normalize: bool = True,
    **kwargs,
):
    """Post-nonlinear dataset with Cauchy noise (CDIM paper)."""
    x, y, z = post_nonlinear_yang2025_ci_fixed(
        sample_size=sample_size,
        conditional_independent=conditional_independent,
        dim_z=dim_z,
        noise="cauchy",
        f1="linear",
        f2="linear",
        noise_scale=noise_scale,
        cond_dep_noise_scale=cond_dep_noise_scale,
        seed=seed,
        normalize=normalize,
    )

    return x, y, z


def post_nonlinear_yang2025_ci_fixed(
    sample_size: int,
    conditional_independent: bool,
    dim_z: int,
    seed: int = None,
    noise: str = "gaussian",
    f1: str = "cos",
    f2: str = "sin",
    noise_scale: float = 0.25,
    cond_dep_noise_scale: float = 0.5,
    normalize: bool = True,
    **kwargs,
):
    """Post-nonlinear dataset (Yang 2025 variant)."""
    if seed is None:
        seed = np.random.randint(0, 10000)
    np.random.seed(seed)
    random.seed(seed)

    dim = dim_z
    funcs = {
        "linear": lambda x: x,
        "square": lambda x: x**2,
        "cos": lambda x: np.cos(x),
        "cube": lambda x: x**3,
        "tanh": lambda x: np.tanh(x),
        "sin": lambda x: np.sin(x),
    }

    if noise == "gaussian":
        sampler = np.random.normal
    elif noise == "laplace":
        sampler = np.random.laplace
    elif noise == "uniform":
        sampler = np.random.uniform
    elif noise == "cauchy":
        sampler = lambda size: np.random.standard_t(df=1, size=size)

    func1 = funcs[f1]
    func2 = funcs[f2]

    noise_x = sampler(size=(sample_size, 1)) * noise_scale
    noise_y = sampler(size=(sample_size, 1)) * noise_scale
    z = np.random.normal(0.0, 1.0, (sample_size, dim))

    z_m = np.mean(z, axis=1).reshape(-1, 1)

    x = z_m + noise_x
    y = z_m + noise_y

    x, y = func1(x), func2(y)

    if not conditional_independent:
        eb = cond_dep_noise_scale * sampler(size=(sample_size, 1))
        x += eb
        y += eb

    x = (x - x.mean(axis=0)) / x.std(axis=0)
    y = (y - y.mean(axis=0)) / y.std(axis=0)
    z = (z - z.mean(axis=0)) / z.std(axis=0)

    return x, y, z


# ======================================================================================================================
# UTILITIES
# ======================================================================================================================


def split_dataset(X: Tensor, Y: Tensor, Z: Tensor, lengths: list = [0.75, 0.15]):
    """Split X,Y, and optionally Z tensors into train, validation, and test."""
    ntrain, nval, ntest = (int(lengths[0] * X.shape[0]), int(lengths[1] * X.shape[0]), int(lengths[2] * X.shape[0]))

    X_train, Y_train = X[:ntrain], Y[:ntrain]
    X_test, Y_test = (
        X[ntrain : ntrain + ntest],
        Y[ntrain : ntrain + ntest],
    )

    if len(lengths) == 2:
        return TensorDataset(X_train, Y_train), TensorDataset(X_test, Y_test)

    X_val, Y_val = (
        X[ntrain + ntest :],
        Y[ntrain + ntest :],
    )

    Z_train = Z[:ntrain]
    Z_test = Z[ntrain : ntrain + ntest]
    Z_val = Z[ntrain + ntest :]

    train_ds = TensorDataset(X_train, Y_train, Z_train)
    val_ds = TensorDataset(X_val, Y_val, Z_val)
    test_ds = TensorDataset(X_test, Y_test, Z_test)

    return train_ds, val_ds, test_ds
