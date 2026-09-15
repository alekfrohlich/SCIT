"""Spectral Independence Test in PyTorch."""

import torch
from omegaconf import DictConfig
from torch import Tensor
from torch.nn import Linear, Module
from torch.nn.utils.parametrizations import spectral_norm
from torch.utils.data import DataLoader

from SCIT.utils import class_from_name, make_mlp


class RepresentationLearningModule(Module):
    def __init__(
        self,
        embedding_x: Module,
        embedding_y: Module,
        d: int,
    ) -> None:
        super().__init__()
        self.U = embedding_x
        self.V = embedding_y

        self.d = d

        # TODO: What if the largest singular value of the operator is considerably smaller than 1?
        # Shoudln't it divide by max(s_1, 1)? Anyhow, this doesn't directly impact orthonormality
        self.M = spectral_norm(Linear(d, d), n_power_iterations=1)

    def forward(self, x: Tensor, y: Tensor) -> tuple:
        u = self.U(x)
        v = self.V(y)

        u_c = u - u.mean(axis=0, keepdim=True)
        v_c = v - v.mean(axis=0, keepdim=True)

        Mv_c = self.M(v_c)

        return u_c, v_c, Mv_c


def off_diag(x: Tensor, y: Tensor):
    n, d = x.shape
    square_term = torch.matmul(x, y.T) ** 2
    return torch.mean(torch.triu(square_term, diagonal=1) + torch.tril(square_term, diagonal=-1)) * n / (n - 1)


def train(
    model: Module,
    train_dl: DataLoader,
    reg_fn: Module,
    reg_strength: float,
    opt: torch.optim.Optimizer,
    logger: callable,
):
    model.train()
    for x, y in train_dl:
        opt.zero_grad()

        u, v, Mv = model(x, y)

        n, d = u.shape

        loss_1 = off_diag(u, Mv)
        loss_2 = -2 * torch.mean(u * Mv) * d

        loss = loss_1 + loss_2

        reg_x = reg_fn(u)
        reg_y = reg_fn(v)

        reg_loss = loss + reg_strength * (reg_x + reg_y)

        reg_loss.backward()
        opt.step()

        logger.log(
            {
                "loss_1/train": loss_1,
                "loss_2/train": loss_2,
                "reg_x/train": reg_x,
                "reg_y/train": reg_y,
                # Composite metrics
                "loss/train": loss,
                "reg_loss/train": reg_loss,
                "norm_M/train": torch.linalg.matrix_norm(model.M.weight, 2),
            }
        )


def validate(
    model: Module,
    val_dl: DataLoader,
    reg_fn: Module,
    reg_strength: float,
    logger: callable,
    epoch: int,
):
    loss_1 = 0
    loss_2 = 0
    reg_x = 0
    reg_y = 0
    model.eval()
    with torch.no_grad():
        for x, y in val_dl:
            u, v, Mv = model(x, y)

            n, d = u.shape

            loss_1 += off_diag(u, Mv)
            loss_2 += -2 * torch.mean(u * Mv) * d

            reg_x += reg_fn(u)
            reg_y += reg_fn(v)

    # Average over the batches
    loss_1 /= len(val_dl)
    loss_2 /= len(val_dl)
    reg_x /= len(val_dl)
    reg_y /= len(val_dl)

    # Composite metrics
    loss = loss_1 + loss_2
    reg_loss = loss + reg_strength * (reg_x + reg_y)

    # Log metrics
    logger.log(
        {
            "loss_1/val": loss_1,
            "loss_2/val": loss_2,
            "reg_x/val": reg_x,
            "reg_y/val": reg_y,
            # Composite metrics
            "loss/val": loss,
            "reg_loss/val": reg_loss,
            "epoch": epoch,
        }
    )
    return reg_loss


def load_model(d: int, seed: int, device: int, cfg: DictConfig):
    scfgs = cfg.experiment_description.split(" ")
    # scfgs[3] = f"d={d}"
    scfgs = " ".join(scfgs) + f" seed={seed}"

    # Load the model and whitening matrices
    load_path = f"{scfgs}/model.pth"
    checkpoint = torch.load(load_path)
    print(f"Model and whitening matrices loaded from {load_path}")

    model_state_dict = checkpoint["model_state_dict"]
    sqrt_cov_u_inv_train = checkpoint["sqrt_cov_u_inv_train"].to(device)
    sqrt_cov_v_inv_train = checkpoint["sqrt_cov_v_inv_train"].to(device)

    U = make_mlp(
        input_dim=cfg.dim_x,
        output_dim=d,
        layer_size=cfg.layer_size,
        n_hidden=cfg.n_hidden,
        activation=class_from_name("torch.nn", cfg.activation),
    ).to(device)
    V = make_mlp(
        input_dim=cfg.dim_y,
        output_dim=d,
        layer_size=cfg.layer_size,
        n_hidden=cfg.n_hidden,
        activation=class_from_name("torch.nn", cfg.activation),
    ).to(device)

    model = RepresentationLearningModule(
        embedding_x=U,
        embedding_y=V,
        d=d,
    ).to(device)

    model.load_state_dict(model_state_dict)

    return (sqrt_cov_u_inv_train, sqrt_cov_v_inv_train, model)
