import os
import sys
import logging
import warnings
from concurrent.futures import ProcessPoolExecutor

import torch
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

# Baselines
from methods.GCIT.GCIT import GCIT
from CCIT import CCIT
from hyppo.conditional import FCIT
from causallearn.utils.cit import CIT
from methods.scetbon import lp_ci_test
from methods.NNLSCIT.nnlscit import lpcmicit
from methods.CDCIT.diffusion_crt import perform_diffusion_crt
from methods.GCM.gcm import GCM_translation
from methods.DGCIT.dgcit import test_dgcit
from methods.PowerKCI.PowerKCIModel import powerKCI_test


sys.path.append(os.path.abspath(".."))  # add repo root to PYTHONPATH

from SCIT.scit import SCIT_Test


# silence all warnings and lower global logging noise in this notebook
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)


########################################################################################################################
# TESTS
########################################################################################################################



'''
# DGCIT parameters
def SCIT_test(x, y, z, alpha):
    test = SCIT_Test(
        output_dim=10,
        n_hidden_u=1,
        layer_size_u=512,
        n_hidden_v=2,
        layer_size_v=512,
        n_hidden_w=3,
        layer_size_w=512,
        lr_inner=0.00723373567764737,
        lr_outer=0.0001599645174348743,
        reg_str_outer=1.9254827416412008,
        reg_str_inner=1.826492113615,
        steps_inner=1,
        max_epochs_warm_inner=100,
        max_epochs=400,
        batch_size=512,
        perc_dim_prune=0.9428801481669404,
        device=0,
    )
    return test.test(x, y, z, alpha=alpha)

'''


## YANG2025 PARAMETERS
def SCIT_test(x, y, z, alpha):
    test = SCIT_Test(
        output_dim=10,
        n_hidden_u=2,
        layer_size_u=128,
        n_hidden_v=2,
        layer_size_v=512,
        n_hidden_w=4,
        layer_size_w=256,
        lr_inner=0.00003051350838147936,
        lr_outer=0.0020703978091406875,
        reg_str_outer=1.9170074640151595,
        reg_str_inner=3.3389076619150533,
        steps_inner=1,
        max_epochs_warm_inner=100,
        max_epochs=400,
        batch_size=512,
        perc_dim_prune=0.937472408591796,
        device=0,
    )
    return test.test(x, y, z, alpha=alpha)
def CDCIT_test(x, y, z, alpha):
    num_samples = x.shape[0]
    div_sample = int(num_samples / 2)

    x_orig = x[:div_sample, ...]
    y_orig = y[:div_sample, ...]
    z_orig = z[:div_sample, ...]

    x_cntrl = x[div_sample:, ...]
    y_cntrl = y[div_sample:, ...]
    z_cntrl = z[div_sample:, ...]

    p_val = perform_diffusion_crt(
        x_orig,
        y_orig,
        z_orig,
        x_cntrl,
        y_cntrl,
        z_cntrl,
        repeat=100,
        device=torch.device("cuda:0"),
        verbose=False,
        seed=444,
        stat="cmi",
        sampling_model="score",
    )

    return p_val < alpha

def DGCIT_test(x, y, z, alpha=0.05):
    pval = test_dgcit(x=x, y=y, z=z)
    return pval < alpha


def NNLSCIT_test(x, y, z, alpha):
    pval = lpcmicit(x, y, z)
    return pval < alpha



def fcit_test(x, y, z, alpha=0.05, seed=42):
    model = DecisionTreeRegressor(random_state=seed)
    stat, pval = FCIT(model=model).test(x, y, z)
    return pval < 0.05


def KCI_test(x, y, z, alpha=0.05):
    cit = CIT(np.column_stack([x, y, z]), method="kci", est_width="median")

    x_indices = list(range(0, x.shape[1]))  # indices of x columns in `data`
    y_indices = list(range(x.shape[1], x.shape[1] + y.shape[1]))  # indices of y columns in `data`
    z_indices = list(range(x.shape[1] + y.shape[1], x.shape[1] + y.shape[1] + z.shape[1]))  # indices of z columns in `data`

    p_value = cit(x_indices, y_indices, z_indices)

    return p_value < alpha

'''
DGCIT parameters
def RCIT_test(x, y, z, alpha=0.05, approx="lpd4", num_f=150, num_f2=20):
    cit = CIT(np.column_stack([x, y, z]), method="rcit", est_width="median", approx=approx, num_f=num_f, num_f2=num_f2)

    z_indices = list(range(2, 2 + z.shape[1]))  # indices of z columns in `data`
    p_value = cit(0, 1, z_indices)

    return p_value < alpha
'''
# yang2025 parameters
def RCIT_test(x, y, z, alpha=0.05, approx="lpd4", num_f=100, num_f2=20):
    cit = CIT(np.column_stack([x, y, z]), method="rcit", est_width="median", approx=approx, num_f=num_f, num_f2=num_f2)

    x_indices = list(range(0, x.shape[1]))  # indices of x columns in `data`
    y_indices = list(range(x.shape[1], x.shape[1] + y.shape[1]))  # indices of y columns in `data`
    z_indices = list(range(x.shape[1] + y.shape[1], x.shape[1] + y.shape[1] + z.shape[1]))  # indices of z columns in `data`
    p_value = cit(x_indices, y_indices, z_indices)

    return p_value < alpha


def CCIT_test(x, y, z, alpha):

    test = CCIT.CCIT(x, y, z, nthread=32)
    return test < alpha


def SCETBON_test(x, y, z, alpha=0.05):
    rank = 1000
    J = 5
    p_norm = 2

    dict_value = lp_ci_test.test_asymptotic_ci(x, y, z, rank, J=J, p_norm=p_norm)

    return not dict_value["H0"]

def GCM_test(x, y, z, alpha=0.05, str_ridge = 1.):
    pval = GCM_translation(X=x, Y=y, Z=z, alpha=alpha, str_ridge=str_ridge)
    return pval < alpha



NAME_TO_TEST = {
    "CDCIT": CDCIT_test,
    "Ours": SCIT_test,
    "GCIT": GCIT,
    "NNLSCIT": NNLSCIT_test,
    "RCIT": RCIT_test,
    "FCIT": fcit_test,
    "KCI": KCI_test,
    "CCIT": CCIT_test,
    "SCETBON": SCETBON_test,
    "GCM": GCM_test,
    'DGCIT': DGCIT_test,
    "PowerKCI": powerKCI_test,
}

########################################################################################################################
# HELPER FUNCTIONS
########################################################################################################################
