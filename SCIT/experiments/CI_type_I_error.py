"""Getting SCIT to control type I error."""

import warnings
from pathlib import Path

import pandas as pd
import scipy
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

import hydra
import wandb

from SCIT.scit import (
    RepresentationLearningModuleInner,
    RepresentationLearningModuleOuter,
    train,
    validate_inner,
    validate_outer,
    whiten_representations,
    validate_statistic,
)
from SCIT.datasets import dataset_ci
from SCIT.utils import EarlyStopper, class_from_name, cross_cov, make_mlp, whitening_matrix

# We are using 'num_workers=0' on purpuse to be able to have the entire dataset in GPU
warnings.filterwarnings("ignore", ".*does not have many workers.*")


def msqrtinvh(m, regularization=1e-6):
    d, d = m.shape
    # Regularize to improve conditioning
    m_reg = (1 - regularization) * m + regularization * torch.eye(d, device=m.device, dtype=m.dtype)

    L, Q = torch.linalg.eigh(m_reg)

    # Filter eigenvalues
    tol = torch.finfo(L.dtype).eps
    sqrt_L_inv = torch.where(L > tol, 1 / L.sqrt(), 0)

    sqrt_cov_x_inv = (Q * sqrt_L_inv) @ Q.T

    return sqrt_cov_x_inv


@hydra.main(
    config_path="config",
    config_name="CI_type_I_error",
    version_base="1.3",
)
def main(cfg: DictConfig):
    # Seed everything
    seed = cfg.seed if cfg.seed >= 0 else torch.randint(high=10000, size=[1]).item()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Update hydra's seed and init wandb
    cfg.seed = seed
    run_cfg = OmegaConf.to_container(cfg, resolve=True)
    wandb.init(project=cfg.project, config=run_cfg)

    # Set machine precision
    torch.set_default_dtype(class_from_name("torch", cfg.precision))
    print(f"Running in {torch.get_default_dtype()} precision")

    # Dataset  =========================================================================================================

    train_ds, val_ds, test_ds, X_train_mean, X_train_std, Y_train_mean, Y_train_std, Z_train_mean, Z_train_std = (
        dataset_ci(
            dataset=cfg.dataset,
            sample_size=cfg.sample_size,
            conditional_independent=cfg.conditional_independent,
            device=cfg.device,
            stack_xz=cfg.stack_xz,
            stack_yz=cfg.stack_yz,
            as_ds=True,
            lengths=(cfg.fraction_train, cfg.fraction_val, cfg.fraction_test),
            dim_z=cfg.dim_z,
        )
    )
    print(f"Dataset sizes: train={len(train_ds)}, val={len(val_ds)}, and test={len(test_ds)}")

    # NOTE: `DataLoader` here is only used to generate random batches for each epoch, as the dataset is already created
    #       inside the GPU
    # NOTE: Shuffling the train dataset makes batches different between epochs. Apparently, it improves training
    train_dl = DataLoader(train_ds, shuffle=True, batch_size=cfg.batch_size)
    val_dl = DataLoader(val_ds, shuffle=False, batch_size=cfg.batch_size)

    # Model  ==========================================================================================================

    U = make_mlp(
        input_dim=cfg.dim_x + cfg.dim_z if cfg.stack_xz else cfg.dim_x,
        output_dim=cfg.output_dim,
        layer_size=cfg.layer_size,
        n_hidden=cfg.n_hidden,
        activation=class_from_name("torch.nn", cfg.activation),
        use_batchnorm=cfg.use_batchnorm,
    ).to(cfg.device)
    V = make_mlp(
        input_dim=cfg.dim_y + cfg.dim_z if cfg.stack_yz else cfg.dim_y,
        output_dim=cfg.output_dim,
        layer_size=cfg.layer_size,
        n_hidden=cfg.n_hidden,
        activation=class_from_name("torch.nn", cfg.activation),
        use_batchnorm=cfg.use_batchnorm,
    ).to(cfg.device)
    W = make_mlp(
        input_dim=cfg.dim_z,
        output_dim=cfg.output_dim,
        layer_size=cfg.layer_size,
        n_hidden=cfg.n_hidden,
        activation=class_from_name("torch.nn", cfg.activation),
        use_batchnorm=cfg.use_batchnorm,
    ).to(cfg.device)

    outer_model = RepresentationLearningModuleOuter(
        embedding_x=U,
        embedding_y=V,
        d=cfg.output_dim,
    ).to(cfg.device)

    inner_model = RepresentationLearningModuleInner(
        embedding_z=W,
        d=cfg.output_dim,
    ).to(cfg.device)

    # Training  ========================================================================================================

    reg_fn = class_from_name("SCIT.utils", cfg.regularizer)
    regularization_strength_outer = cfg.reg_str_outer
    regularization_strength_inner = cfg.reg_str_inner
    opt_outer = torch.optim.Adam(outer_model.parameters(), lr=cfg.lr_outer)
    opt_inner = torch.optim.Adam(inner_model.parameters(), lr=cfg.lr_inner)

    # NOTE: The inner model is training for some time before starting to train the outer model. This can be seen as a
    #       hyperparameter
    early_stopper = EarlyStopper(patience=5, min_delta=cfg.min_delta)
    epoch_inner_stopped = cfg.max_epochs_warm_inner
    for epoch in tqdm(range(cfg.max_epochs_warm_inner), desc="warm start"):
        validation_epoch = epoch % 5 == 0
        train(
            outer_model=outer_model,
            inner_model=inner_model,
            train_dl=train_dl,
            reg_fn=reg_fn,
            regularization_strength_outer=regularization_strength_outer,
            regularization_strength_inner=regularization_strength_inner,
            opt_outer=opt_outer,
            opt_inner=opt_inner,
            logger=wandb,
            freeze_outer=True,
        )
        if validation_epoch and early_stopper.early_stop(
            validate_inner(
                outer_model=outer_model,
                inner_model=inner_model,
                val_dl=val_dl,
                reg_fn=reg_fn,
                regularization_strength_inner=regularization_strength_inner,
                logger=wandb,
                epoch=epoch,
            )
        ):
            epoch_inner_stopped = epoch
            print("Early stopping warm start at epoch={}".format(epoch))
            break

    early_stopper = EarlyStopper(patience=cfg.patience, min_delta=cfg.min_delta)
    for epoch in tqdm(
        range(1 + epoch_inner_stopped, 1 + epoch_inner_stopped + cfg.max_epochs), desc="train (seed={})".format(seed)
    ):
        validation_epoch = epoch % cfg.check_val_every_n_epoch == 0
        train(
            outer_model=outer_model,
            inner_model=inner_model,
            train_dl=train_dl,
            reg_fn=reg_fn,
            regularization_strength_outer=regularization_strength_outer,
            regularization_strength_inner=regularization_strength_inner,
            opt_outer=opt_outer,
            opt_inner=opt_inner,
            logger=wandb,
            freeze_outer=False,
            steps_inner=cfg.steps_inner,
        )

        if validation_epoch:
            validate_inner(
                outer_model=outer_model,
                inner_model=inner_model,
                val_dl=val_dl,
                reg_fn=reg_fn,
                regularization_strength_inner=regularization_strength_inner,
                logger=wandb,
                epoch=epoch,
            )
            if early_stopper.early_stop(
                validate_outer(
                    outer_model=outer_model,
                    inner_model=inner_model,
                    val_dl=val_dl,
                    reg_fn=reg_fn,
                    regularization_strength_outer=regularization_strength_outer,
                    logger=wandb,
                    epoch=epoch,
                )
            ):
                print("Early stopping at epoch={}".format(epoch))
                break

            validate_statistic(
                outer_model=outer_model,
                inner_model=inner_model,
                val_ds=val_ds,
                logger=wandb,
                epoch=epoch,
                output_dim=cfg.output_dim,
                reg_whitening=cfg.reg_whitening,
                alpha=cfg.alpha,
            )

    # Whitening  =======================================================================================================

    # NOTE: If thinking of the learned features as functions in a Hilbert space of functions, this is not changing their
    #       span, but only (approximately) enforcing that they are orthonormal. Notice this is complementary to the
    #       regularization term of the loss
    # NOTE: In deep learning terms, whitening is trying to make the features uncorrelated and of unit variance. That is,
    #       to have covariance matrix equal to identity matrix.
    print("Whitening representations on train set...")
    x_train, y_train, z_train = train_ds.tensors
    outer_model.eval()
    inner_model.eval()
    with torch.inference_mode():
        u_c, v_c, _ = outer_model(x_train, y_train, z_train)
        w_c, __ = inner_model(x_train, y_train, z_train)
        u_c, v_c, w_c = u_c.double(), v_c.double(), w_c.double()

        wandb.log(
            {
                "cond_cov_u/train": torch.linalg.cond(cross_cov(u_c, u_c), p=2),
                "cond_cov_v/train": torch.linalg.cond(cross_cov(v_c, v_c), p=2),
                "cond_cov_w/train": torch.linalg.cond(cross_cov(w_c, w_c), p=2),
            }
        )

        sqrt_cov_u_inv_train, sqrt_cov_v_inv_train, sqrt_cov_w_inv_train = whiten_representations(
            u=u_c, v=v_c, w=w_c, regularization=cfg.reg_whitening
        )

    # Testing Hypothesis  ==============================================================================================

    print("Computing test statistic...")
    x_test, y_test, z_test = test_ds.tensors
    outer_model.eval()
    inner_model.eval()
    with torch.inference_mode():
        u_c, v_c, _ = outer_model(x_test, y_test, z_test)
        w_c, __ = inner_model(x_test, y_test, z_test)
        u_c, v_c, w_c = u_c.double(), v_c.double(), w_c.double()

        n, d = u_c.shape
        r = cfg.repeats

        # Whiten features
        u_hat = u_c @ sqrt_cov_u_inv_train
        v_hat = v_c @ sqrt_cov_v_inv_train
        w_hat = w_c @ sqrt_cov_w_inv_train

        # # Log orthonormality of learned features before and after whitening
        Id = torch.eye(d, device=u_c.device, dtype=u_c.dtype)
        wandb.log(
            {
                "||C_util - Id||_op/test": torch.linalg.matrix_norm(cross_cov(u_c, u_c) - Id, ord=2),
                "||C_vtil - Id||_op/test": torch.linalg.matrix_norm(cross_cov(v_c, v_c) - Id, ord=2),
                "||C_wtil - Id||_op/test": torch.linalg.matrix_norm(cross_cov(w_c, w_c) - Id, ord=2),
                "||C_uhat - Id||_op/test": torch.linalg.matrix_norm(cross_cov(u_hat, u_hat) - Id, ord=2),
                "||C_vhat - Id||_op/test": torch.linalg.matrix_norm(cross_cov(v_hat, v_hat) - Id, ord=2),
                "||C_what - Id||_op/test": torch.linalg.matrix_norm(cross_cov(w_hat, w_hat) - Id, ord=2),
            }
        )
        u_c = u_hat
        v_c = v_hat
        w_c = w_hat

        print("Computing test statistic...")

        cov_u = cross_cov(u_c, u_c)
        cov_v = cross_cov(v_c, v_c)
        cov_w = cross_cov(w_c, w_c)
        cov_uv = cross_cov(u_c, v_c)
        cov_uw = cross_cov(u_c, w_c)
        cov_wv = cross_cov(w_c, v_c)
        cov_w_inv = whitening_matrix(w_c, sqrt=False)

        # M = cov_uv - cov_uw @ cov_wv
        Q_diag = 1.0 - ((w_c @ whitening_matrix(w_c)) ** 2).sum(dim=1, keepdim=True) / (n - 1)
        M = cross_cov(Q_diag * u_c, v_c)
        test_stat_biased = len(test_ds) * torch.linalg.norm(M, ord="fro") ** 2

        # Error matrix
        E = torch.linalg.lstsq(cov_w, Id - cov_w).solution
        bias = len(test_ds) * (
            2 * torch.trace(M @ (cov_uw @ E @ cov_wv).T) - torch.linalg.norm(cov_uw @ E @ cov_wv, ord="fro") ** 2
        )

        test_stat = test_stat_biased - bias

        # NOTE: The significance of the test statistic observed on the test set is determined in two ways:
        #       (1) by using the 95% quantile of the theoretical null distribution (chi-squared with cfg.output_dim
        #           degrees of freedom)
        #       (2) by using an intermediate theoretical result that uses the eigenvalues of the covariance matrices of
        #           the features in combination with a sample of (cfg.output_dim)^2 standard normal random variables to
        #           estimate the null distribution. This gives a p-value that can be used for testing
        print("Computing p-value...")

        evals_cov_u = torch.linalg.eigvalsh(cov_u).expand(1, d)
        evals_cov_v = torch.linalg.eigvalsh(cov_v).expand(1, d)

        evals_uv = evals_cov_u.T @ evals_cov_v
        chi2 = torch.normal(mean=0.0, std=1.0, size=(1000, d, d), device=u_c.device, dtype=u_c.dtype) ** 2
        test_stat_null_clt = (evals_uv * chi2).sum(dim=(1, 2))
        print(f"Simulated Test Stat shape: {test_stat_null_clt.shape}")

        # TODO: Here I could do permutation, bootstrap or other empirical test over the test dataset if analytical null
        #       distribution is too hard to describe with theory.

        # Partial correlation
        if cfg.output_dim == 1:
            # https://pingouin-stats.org/build/html/generated/pingouin.partial_corr.html
            from pingouin import partial_corr

            # compute partial correlation between u_c and v_c given w_c
            df_pc = pd.DataFrame(
                {
                    "u_c": u_c.squeeze().cpu().numpy(),
                    "v_c": v_c.squeeze().cpu().numpy(),
                    "w_c": w_c.squeeze().cpu().numpy(),
                }
            )
            pc_result = partial_corr(data=df_pc, x="u_c", y="v_c", covar=["w_c"], method="pearson")
            # wandb.log(
            #     {
            #         "partial_correlation": pc_result["r"].values[0],
            #         "partial_correlation_pval": pc_result["p-val"].values[0],
            #     }
            # )

        metrics = {
            # "cmi": torch.log(torch.clamp(1 + (u_c * torch.diag() @ v_c).sum(dim=1), min=1e-6)).sum(),
            "test_stat": test_stat.item(),
            "test_stat_no_norm": test_stat / len(test_ds),
            "bias": bias.item(),
            "q95": torch.quantile(test_stat_null_clt, q=1 - cfg.alpha).item(),
            "p_val_hs": (test_stat_null_clt >= test_stat).type_as(test_stat).mean().item(),
            "reject_chi2": test_stat.item() > scipy.stats.chi2.ppf(1 - cfg.alpha, df=cfg.output_dim**2),
            "reject_resampled_chi2": (test_stat_null_clt >= test_stat).type_as(test_stat).mean().item() < cfg.alpha,
            # "partial_correlation": pc_result["r"].values[0],
            # "partial_correlation_pval": pc_result["p-val"].values[0],
        }
        wandb.log(metrics)

    # Save metrics and models
    EXPERIMENT_PATH = Path(f"hydra/{cfg.project}/{cfg.experiment_description} seed={seed}")
    EXPERIMENT_PATH.mkdir(parents=True, exist_ok=True)

    if cfg.save_model:
        MODEL_PATH = EXPERIMENT_PATH / "model.pth"
        torch.save(
            {
                "outer_model_state_dict": outer_model.state_dict(),
                "inner_model_state_dict": inner_model.state_dict(),
                "u_state_dict": U.state_dict(),
                "v_state_dict": V.state_dict(),
                "w_state_dict": W.state_dict(),
                "sqrt_cov_u_inv_train": sqrt_cov_u_inv_train,
                "sqrt_cov_v_inv_train": sqrt_cov_v_inv_train,
                "sqrt_cov_w_inv_train": sqrt_cov_w_inv_train,
                "X_train_mean": X_train_mean,
                "X_train_std": X_train_std,
                "Y_train_mean": Y_train_mean,
                "Y_train_std": Y_train_std,
            },
            MODEL_PATH,
        )
        print(f"Model and whitening matrices saved to {MODEL_PATH}")

    METRICS_PATH = EXPERIMENT_PATH / "metrics.csv"
    metrics |= run_cfg
    try:
        df = pd.read_csv(METRICS_PATH)
        df = pd.concat([df, pd.DataFrame(data=metrics, index=[0])], ignore_index=True)
        df.to_csv(METRICS_PATH, index=False)
    except FileNotFoundError:
        pd.DataFrame(data=metrics, index=[0]).to_csv(METRICS_PATH, index=False)


if __name__ == "__main__":
    main()
