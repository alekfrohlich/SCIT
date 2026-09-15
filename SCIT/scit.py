"""SCIT in PyTorch."""

import os
import time

import torch
import torch.nn.utils.parametrize as parametrize
from torch import Tensor
from torch.nn import Linear, Module
from torch.nn.utils.parametrizations import spectral_norm
from torch.utils.data import DataLoader

from SCIT.utils import Symmetric, cross_cov, whitening_matrix, make_mlp, orthonormal_logfro_reg, set_seed, class_from_name
from SCIT.datasets import split_dataset
from SCIT import suit
import scipy
import numpy as np
from tqdm import tqdm


# TODO:
# Remove logger, instead return metrics dicts
# Move from train to train_step
# Refactor data pipeline by removing standardization from split_dataset




class SCIT_Test(object):
    def __init__(
        self,
        output_dim: int,
        n_hidden_u: int,
        layer_size_u: int,
        n_hidden_w: int,
        layer_size_w: int,
        n_hidden_v: int,
        layer_size_v: int,
        lr_inner: float,
        lr_outer: float,
        reg_str_inner: float,
        reg_str_outer: float,
        steps_inner: int,
        batch_size: int,
        max_epochs: int,
        max_epochs_warm_inner: int,
        perc_dim_prune: float,
        device=None,
    ):
        self.output_dim = output_dim
        self.n_hidden_v = n_hidden_v
        self.layer_size_v = layer_size_v
        self.n_hidden_w = n_hidden_w
        self.layer_size_w = layer_size_w
        self.n_hidden_u = n_hidden_u
        self.layer_size_u = layer_size_u
        self.lr_inner = lr_inner
        self.lr_outer = lr_outer
        self.reg_str_inner = reg_str_inner
        self.reg_str_outer = reg_str_outer
        self.steps_inner = steps_inner
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.max_epochs_warm_inner = max_epochs_warm_inner
        self.perc_dim_prune = perc_dim_prune

        if device is None:
            self.device = "cuda" if self.torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.regularizer = orthonormal_logfro_reg

    def test(self, x: Tensor, y: Tensor, z: Tensor, alpha: float) -> bool:

        set_seed()
        if type(x) is np.ndarray:
            x = torch.from_numpy(x).to(self.device).to(torch.float32)
        if type(y) is np.ndarray:
            y = torch.from_numpy(y).to(self.device).to(torch.float32)
        if type(z) is np.ndarray:
            z = torch.from_numpy(z).to(self.device).to(torch.float32)

        dim_x, dim_y, dim_z = x.shape[1], y.shape[1], z.shape[1]
        y = torch.hstack((y, z))

        train_ds, val_ds, test_ds = split_dataset(x, y, z, lengths=[0.8, 0.002, 0.198])
        train_dl = DataLoader(train_ds, shuffle=True, batch_size=self.batch_size)

        U = make_mlp(
            input_dim=dim_x,
            output_dim=self.output_dim,
            layer_size=self.layer_size_u,
            n_hidden=self.n_hidden_u,
            activation=torch.nn.Tanh,
            use_batchnorm=False,
        ).to(self.device)
        V = make_mlp(
            input_dim=dim_y + dim_z,
            output_dim=self.output_dim,
            layer_size=self.layer_size_v,
            n_hidden=self.n_hidden_v,
            activation=torch.nn.Tanh,
            use_batchnorm=False,
        ).to(self.device)
        W = make_mlp(
            input_dim=dim_z,
            output_dim=2*self.output_dim,
            layer_size=self.layer_size_w,
            n_hidden=self.n_hidden_w,
            activation=torch.nn.Tanh,
            use_batchnorm=False,
        ).to(self.device)

        outer_model = RepresentationLearningModuleOuter(
            embedding_x=U,
            embedding_y=V,
            d=self.output_dim,
        ).to(self.device)

        inner_model = RepresentationLearningModuleInner(
            embedding_z=W,
            d=2*self.output_dim,
        ).to(self.device)

        opt_outer = torch.optim.Adam(outer_model.parameters(), lr=self.lr_outer)
        opt_inner = torch.optim.Adam(inner_model.parameters(), lr=self.lr_inner)

        for epoch in range(self.max_epochs_warm_inner):
            train(
                outer_model=outer_model,
                inner_model=inner_model,
                train_dl=train_dl,
                reg_fn=self.regularizer,
                regularization_strength_outer=self.reg_str_outer,
                regularization_strength_inner=self.reg_str_inner,
                opt_outer=opt_outer,
                opt_inner=opt_inner,
                logger=None,
                freeze_outer=True,
            )

        for epoch in range(1 + self.max_epochs_warm_inner, 1 + self.max_epochs_warm_inner + self.max_epochs):
            train(
                outer_model=outer_model,
                inner_model=inner_model,
                train_dl=train_dl,
                reg_fn=self.regularizer,
                regularization_strength_outer=self.reg_str_outer,
                regularization_strength_inner=self.reg_str_inner,
                opt_outer=opt_outer,
                opt_inner=opt_inner,
                logger=None,
                freeze_outer=False,
                steps_inner=self.steps_inner,
            )

        x_train, y_train, z_train = train_ds.tensors
        outer_model.eval()
        inner_model.eval()
        with torch.inference_mode():
            u_c, v_c, _ = outer_model(x_train, y_train, z_train)
            w_c, __ = inner_model(x_train, y_train, z_train)
            u_c, v_c, w_c = u_c.double(), v_c.double(), w_c.double()

            sqrt_cov_u_inv_train, sqrt_cov_v_inv_train, sqrt_cov_w_inv_train = whiten_representations(
                u=u_c, v=v_c, w=w_c
            )

        x_test, y_test, z_test = test_ds.tensors

        outer_model.eval()
        inner_model.eval()
        with torch.inference_mode():
            u_c, v_c, _ = outer_model(x_test, y_test, z_test)
            w_c, __ = inner_model(x_test, y_test, z_test)
            u_c, v_c, w_c = u_c.double(), v_c.double(), w_c.double()

            n, d = u_c.shape

            # Whiten features
            u_c = u_c @ sqrt_cov_u_inv_train
            v_c = v_c @ sqrt_cov_v_inv_train
            w_c = w_c @ sqrt_cov_w_inv_train

            Q_diag = 1.0 - ((w_c @ whitening_matrix(w_c)) ** 2).sum(dim=1, keepdim=True) / (n - 1)

            dim_prune = max(int(self.perc_dim_prune * self.output_dim),1)

            M = cross_cov(Q_diag * u_c, v_c)
            _, singular_vals, _ = torch.linalg.svd(M, full_matrices=False)
            prunned_singular_vals = singular_vals[:dim_prune]
            test_stat = (len(test_ds) * (prunned_singular_vals ** 2)).sum()


            reject_chi2 = test_stat.item() > scipy.stats.chi2.ppf(1 - alpha, df=dim_prune**2)

            return reject_chi2


class RepresentationLearningModuleOuter(Module):
    def __init__(
        self,
        embedding_x: Module,
        embedding_y: Module,
        d: int,
        d_v: int = None,
    ) -> None:
        super().__init__()
        self.U = embedding_x
        self.V = embedding_y

        self.d = d
        d_v_actual = d_v if d_v is not None else d

        # TODO: What if the largest singular value of the operator is considerably smaller than 1?
        # Shoudln't it divide by max(s_1, 1)? Anyhow, this doesn't directly impact orthonormality
        self.M = spectral_norm(Linear(d_v_actual, d, bias=False), n_power_iterations=1)
        torch.nn.init.xavier_uniform_(self.M.weight)   # gain = 1.0



    def forward(self, x: Tensor, yz: Tensor, z: Tensor) -> tuple:
        u = self.U(x)
        v = self.V(yz)

        u_c = u - u.mean(axis=0, keepdim=True)
        v_c = v - v.mean(axis=0, keepdim=True)

        Mv_c = self.M(v_c)

        return u_c, v_c, Mv_c


class RepresentationLearningModuleInner(Module):
    def __init__(
        self,
        embedding_z: Module,
        d: int,
    ) -> None:
        super().__init__()
        self.W = embedding_z

        self.d = d

        linear_layer = Linear(d, d, bias=False)
        parametrize.register_parametrization(linear_layer, "weight", Symmetric())
        self.N = spectral_norm(linear_layer, n_power_iterations=1)
        torch.nn.init.xavier_uniform_(self.N.weight)   # gain = 1.0

    def forward(self, x: Tensor, yz: Tensor, z: Tensor) -> tuple:
        w = self.W(z)
        w_c = w - w.mean(axis=0, keepdim=True)

        Nw_c = self.N(w_c)

        return w_c, Nw_c


def off_diag(x: Tensor, y: Tensor) -> Tensor:
    npts, _ = x.shape
    square_term = torch.matmul(x, y.T) ** 2
    off_diag = (
        torch.mean(torch.triu(square_term, diagonal=1) + torch.tril(square_term, diagonal=-1)) * npts / (npts - 1)
    )
    return off_diag


def train(
    outer_model: RepresentationLearningModuleOuter,
    inner_model: RepresentationLearningModuleInner,
    train_dl: DataLoader,
    reg_fn: callable,
    regularization_strength_outer: float,
    regularization_strength_inner: float,
    opt_outer: torch.optim.Optimizer,
    opt_inner: torch.optim.Optimizer,
    logger: callable = None,
    freeze_outer: bool = False,
    steps_inner: int = 10,
):
    running_loss = 0.0
    updates = 0

    for batch, (x, yz, z) in enumerate(train_dl):
        # Defensive check: handle NaN inputs
        x = torch.nan_to_num(x)
        yz = torch.nan_to_num(yz)
        z = torch.nan_to_num(z)
        
        # Outer optimization=======================================================================
        if batch % steps_inner == steps_inner - 1 and not freeze_outer:
            outer_model.train()
            inner_model.eval()
            opt_outer.zero_grad()

            u_c, v_c, Mv_c = outer_model(x, yz, z)
            with torch.no_grad():
                w_c, _ = inner_model(x, yz, z)

            n, d = u_c.shape

            # TODO: Least squares
            t1 = off_diag(u_c, Mv_c)
            t2 = -2 * torch.mean(u_c * Mv_c) * d
            t3 = (
                        2
                        * torch.mean(torch.matmul(u_c, Mv_c.T) * torch.matmul(w_c, w_c.T))
                        * n
                        / (n - 1)
                    )

            reg_x = reg_fn(u_c)
            reg_y = reg_fn(v_c)

            # Composite metrics
            ncp_loss = t1 + t2
            outer_loss = ncp_loss + t3
            reg_outer_loss = outer_loss + regularization_strength_outer * (reg_x + reg_y)

            # Skip batch if loss is invalid
            if not torch.isfinite(reg_outer_loss):
                continue

            reg_outer_loss.backward()
            torch.nn.utils.clip_grad_norm_(outer_model.parameters(), max_norm=5.0)
            opt_outer.step()
            running_loss += reg_outer_loss.detach().item()
            updates += 1

            if logger is not None:
                logger.log(
                    {
                        "outer_loss_1/train": t1,
                        "outer_loss_2/train": t2,
                        "outer_loss_3/train": t3,
                        "reg_x/train": reg_x,
                        "reg_y/train": reg_y,
                        # Composite metrics
                        "outer_loss_12/train": ncp_loss,
                        "outer_loss_23/train": t2 + t3,
                        "outer_loss/train": outer_loss,
                        "reg_outer_loss/train": reg_outer_loss,
                        "norm_M/train": torch.linalg.matrix_norm(outer_model.M.weight, 2),
                    }
                )

        # Inner optimization=======================================================================
        else:
            outer_model.eval()
            inner_model.train()
            opt_inner.zero_grad()

            with torch.no_grad():
                u_c, v_c, Mv_c = outer_model(x, yz, z)
            w_c, Nw_c = inner_model(x, yz, z)

            n, d = u_c.shape

            inner_loss_1 = off_diag(w_c, Nw_c)
            inner_loss_2 = (
                -2
                * torch.mean(torch.matmul(u_c, Mv_c.T) * torch.matmul(w_c, Nw_c.T))
                * n
                / (n - 1)
            )
            inner_loss = inner_loss_1 + inner_loss_2

            reg_z = reg_fn(w_c)

            reg_inner_loss = inner_loss + regularization_strength_inner * reg_z

            # Skip batch if loss is invalid
            if not torch.isfinite(reg_inner_loss):
                continue

            reg_inner_loss.backward()
            torch.nn.utils.clip_grad_norm_(inner_model.parameters(), max_norm=5.0)
            opt_inner.step()
            running_loss += reg_inner_loss.detach().item()
            updates += 1

            if logger is not None:
                logger.log(
                    {
                        "inner_loss_1/train": inner_loss_1,
                        "inner_loss_2/train": inner_loss_2,
                        "reg_z/train": reg_z,
                        # Composite metrics
                        "inner_loss/train": inner_loss,
                        "reg_inner_loss/train": reg_inner_loss,
                        "norm_N/train": torch.linalg.matrix_norm(inner_model.N.weight, 2),
                        "N_symm/train": torch.dist(inner_model.N.weight, inner_model.N.weight.T),
                    }
                )

    if updates == 0:
        return float("inf")

    return running_loss / updates


def validate_statistic(
    outer_model: RepresentationLearningModuleOuter,
    inner_model: RepresentationLearningModuleInner,
    val_ds: DataLoader,
    output_dim: int,
    epoch: int,
    logger: callable = None,
    reg_whitening: float = 1e-6,
    alpha: float = 0.05,
):
    x_val, yz_val, z_val = val_ds.tensors
    outer_model.eval()
    inner_model.eval()

    with torch.inference_mode():
        u_c, v_c, _ = outer_model(x_val, yz_val, z_val)
        w_c, __ = inner_model(x_val, yz_val, z_val)
        u_c, v_c, w_c = u_c.double(), v_c.double(), w_c.double()

        sqrt_cov_u_inv, sqrt_cov_v_inv, sqrt_cov_w_inv = whiten_representations(
            u=u_c, v=v_c, w=w_c, regularization=reg_whitening
        )

        u_c = u_c @ sqrt_cov_u_inv
        v_c = v_c @ sqrt_cov_v_inv
        w_c = w_c @ sqrt_cov_w_inv

        cov_uv = cross_cov(u_c, v_c)
        cov_uw = cross_cov(u_c, w_c)
        cov_wv = cross_cov(w_c, v_c)
        cov_w = cross_cov(w_c, w_c)

        M = cov_uv - cov_uw @ cov_wv

        number_samples = len(val_ds)
        test_stat_biased_no_norm = torch.linalg.norm(M, ord="fro") ** 2
        test_stat_biased = test_stat_biased_no_norm * number_samples

        n, d = u_c.shape
        Id = torch.eye(d, device=u_c.device, dtype=u_c.dtype)

        E = torch.linalg.lstsq(cov_w, Id - cov_w).solution
        bias = number_samples * (
            2 * torch.trace(M @ (cov_uw @ E @ cov_wv).T) - torch.linalg.norm(cov_uw @ E @ cov_wv, ord="fro") ** 2
        )

        test_stat = test_stat_biased - bias

    threshold_test_stat = scipy.stats.chi2.ppf(1 - alpha, df=output_dim**2)

    diff = test_stat - threshold_test_stat

    if logger is not None:
        logger.log(
            {
                "stat_test/test_stat": test_stat,
                "stat_test/threshold_test_stat": threshold_test_stat,
                "stat_test/test_stat_biased": test_stat_biased,
                "stat_test/bias": bias,
                "stat_test/epoch": epoch,
                "stat_test/test_stat_biased_no_norm": test_stat_biased_no_norm,
                "stat_test/diff": diff,
            }
        )


def validate_outer(
    outer_model: RepresentationLearningModuleOuter,
    inner_model: RepresentationLearningModuleInner,
    val_dl: DataLoader,
    reg_fn: callable,
    regularization_strength_outer: float,
    epoch: int,
    logger: callable = None,
):
    reg_x = 0
    reg_y = 0
    t1 = 0
    t2 = 0
    t3 = 0
    outer_model.eval()
    inner_model.eval()
    with torch.no_grad():
        for batch, (x, yz, z) in enumerate(val_dl):
            u_c, v_c, Mv_c = outer_model(x, yz, z)
            w_c, Nw_c = inner_model(x, yz, z)

            n, d = u_c.shape

            cov_uw = cross_cov(u_c, w_c)
            cov_wv = cross_cov(w_c, v_c)
            cov_w_inv = whitening_matrix(w_c, sqrt=False)

            t1 += off_diag(u_c, Mv_c)
            t2 += -2 * torch.mean(u_c * Mv_c) * d
            t3 += 2 * (cov_uw @ cov_w_inv @ cov_wv @ outer_model.M.weight.T).trace()

            reg_x += reg_fn(u_c)
            reg_y += reg_fn(v_c)

    # Average over the batches
    t1 = t1 / len(val_dl)
    t2 = t2 / len(val_dl)
    t3 = t3 / len(val_dl)
    reg_x = reg_x / len(val_dl)
    reg_y = reg_y / len(val_dl)

    # Composite metrics
    t12 = t1 + t2
    t23 = t2 + t3
    outer_loss = t1 + t2 + t3
    reg_outer_loss = outer_loss + regularization_strength_outer * (reg_x + reg_y)

    # Log metrics
    if logger is not None:
        logger.log(
            {
                "outer_loss_1/val": t1,
                "outer_loss_2/val": t2,
                "outer_loss_3/val": t3,
                "reg_x/val": reg_x,
                "reg_y/val": reg_y,
                # Composite metrics
                "outer_loss_12/val": t12,
                "outer_loss_23/val": t23,
                "outer_loss/val": outer_loss,
                "reg_outer_loss/val": reg_outer_loss,
                "epoch": epoch,
            }
        )
    return reg_outer_loss


def validate_inner(
    outer_model: RepresentationLearningModuleOuter,
    inner_model: RepresentationLearningModuleInner,
    val_dl: DataLoader,
    reg_fn: callable,
    regularization_strength_inner: float,
    epoch: int,
    logger: callable = None,
):
    inner_loss_1 = 0
    inner_loss_2 = 0
    reg_z = 0
    outer_model.eval()
    inner_model.eval()
    with torch.no_grad():
        for batch, (x, yz, z) in enumerate(val_dl):
            u_c, v_c, Mv_c = outer_model(x, yz, z)
            w_c, Nw_c = inner_model(x, yz, z)

            n, d = u_c.shape

            inner_loss_1 += off_diag(w_c, Nw_c)
            inner_loss_2 += (
                -2
                * torch.mean(torch.matmul(u_c, Mv_c.T) * torch.matmul(w_c, Nw_c.T))
                * n
                / (n - 1)
            )

            reg_z += reg_fn(w_c)

    inner_loss_1 = inner_loss_1 / len(val_dl)
    inner_loss_2 = inner_loss_2 / len(val_dl)
    reg_z = reg_z / len(val_dl)
    inner_loss = inner_loss_1 + inner_loss_2
    reg_inner_loss = inner_loss + regularization_strength_inner * reg_z

    if logger is not None:
        logger.log(
            {
                "inner_loss_1/val": inner_loss_1,
                "inner_loss_2/val": inner_loss_2,
                "reg_z/val": reg_z,
                # Composite metrics
                "inner_loss/val": inner_loss,
                "reg_inner_loss/val": reg_inner_loss,
                "epoch": epoch,
            }
        )
    return reg_inner_loss


def whiten_representations(
    u: Tensor,
    v: Tensor,
    w: Tensor,
    regularization: float = 1e-6,
    # logger: callable,
) -> Tensor:
    """Whiten representations of u, v, and w."""
    sqrt_cov_u_inv = whitening_matrix(u, regularization=regularization)
    sqrt_cov_v_inv = whitening_matrix(v, regularization=regularization)
    sqrt_cov_w_inv = whitening_matrix(w, regularization=regularization)

    return sqrt_cov_u_inv, sqrt_cov_v_inv, sqrt_cov_w_inv
