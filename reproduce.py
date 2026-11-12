import argparse
import fcntl
import logging
import multiprocessing as mp
import os
import sys
import warnings

import numpy as np
import torch
from tqdm import tqdm

sys.path.append(os.path.abspath(".."))  # add repo root to PYTHONPATH

from SCIT.datasets import dataset_ci

# silence all warnings and lower global logging noise in this notebook
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)
import time

DIMENSIONS_Z = [50, 100, 150, 200, 250, 300]
REPEATS = 100
SAMPLE_SIZE = 1000
ALPHA = 0.05
NUMBER_RUNS = 5
CI = False


DATASET_CHOICES = [
    "he2025_ci",
    "benchmark_rebuttal_ci",
    "post_nonlinear_yang2025_ci",
    "post_nonlinear_cauchy_ci",
    "postnonlinear_nonsmooth_test_ci",
    "post_nonlinear_ren_2025_ci",
    "multidim_linear_ci",
    "linear_cauchy_ci",
    "post_nonlinear_dgcit_CI",
]
from SCIT.test_methods import NAME_TO_TEST


def run_single_repeat(i, method, sample_size, dim_z, alpha, CI, gpu_id, dataset_name):
    # Bind this process to a GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    torch.cuda.set_device(0)

    x, y, z = dataset_ci(
        dataset=dataset_name,
        sample_size=sample_size,
        conditional_independent=CI,
        device=0,
        stack_xz=False,
        stack_yz=False,
        as_ds=False,
        lengths=[0.8, 0.1, 0.1],
        seed=i,
        dim_z=dim_z,
    )

    # Benchmark methods operate on numpy arrays.
    x = x.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    z = z.detach().cpu().numpy()

    print(f"Repeat {i}: Running on GPU {gpu_id} for dim_z={dim_z}, dataset={dataset_name}")

    start_time = time.time()
    result = method(x, y, z, alpha)

    end_time = time.time()
    time_taken = end_time - start_time
    return int(result), time_taken


def return_reject_hypothesis_parallel(
    method,
    num_run,
    repeats=100,
    sample_size=500,
    dim_z=1,
    alpha=0.05,
    CI=True,
    workers_per_gpu=5,
    dataset_name="he2025_ci",
):
    n_gpus = torch.cuda.device_count()
    total_workers = n_gpus * workers_per_gpu

    # GPU assignment: repeat i → GPU (i % n_gpus)
    gpu_ids = [i % n_gpus for i in range(repeats)]

    # Build chunk args
    tasks = [(i, gpu_ids[i]) for i in range(repeats)]

    with mp.Pool(total_workers) as pool:
        results = []
        for i, gpu_id in tqdm(tasks):
            results.append(
                pool.apply_async(
                    run_single_repeat,
                    args=(i + repeats * num_run, method, sample_size, dim_z, alpha, CI, gpu_id, dataset_name),
                )
            )

        results = [r.get() for r in results]

    array = np.array(results)

    mean_score, mean_time = array.mean(axis=0)
    std_score, std_time = array.std(axis=0)

    return mean_score, std_score, mean_time, std_time


def append_line_locked(path, line):
    """Append line to path using an exclusive advisory lock (POSIX)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        # lock, write, flush, fsync, unlock
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(line)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass
        fcntl.flock(f, fcntl.LOCK_UN)


DATASET_CHOICES = [
    "he2025_ci",
    "benchmark_rebuttal_ci",
    "post_nonlinear_yang2025_ci",
    "post_nonlinear_cauchy_ci",
    "postnonlinear_nonsmooth_test_ci",
    "post_nonlinear_ren_2025_ci",
    "multidim_linear_ci",
    "linear_cauchy_ci",
    "postnonlinear_dgcit",
]

if __name__ == "__main__":
    # TODO: Add argparse for selecting between experiments
    method_name = "KCI"
    class_name_to_test = NAME_TO_TEST[method_name]
    dataset_name = "postnonlinear_dgcit"

    list_reject_hypothesis = []
    for dim_z in DIMENSIONS_Z:
        for id_run in range(NUMBER_RUNS):
            reject_hypothesis_mean, reject_hypothesis_std, time_mean, time_std = return_reject_hypothesis_parallel(
                class_name_to_test,
                num_run=id_run,
                repeats=REPEATS,
                sample_size=SAMPLE_SIZE,
                dim_z=dim_z,
                alpha=ALPHA,
                CI=CI,
                workers_per_gpu=10,
                dataset_name=dataset_name,
            )
            list_reject_hypothesis.append(reject_hypothesis_mean)

        reject_hypothesis_mean = np.mean(list_reject_hypothesis)
        reject_hypothesis_std = np.std(list_reject_hypothesis)

        log_path = f"logs_bench_test_sample/single_{method_name}_dim_z_benchmark_{'type_1' if CI else 'Power'}.txt"
        append_line_locked(
            log_path,
            f"dim_z: {dim_z}, Reject Hypothesis Mean: {reject_hypothesis_mean}, Reject Hypothesis Std: {reject_hypothesis_std}, Time Mean: {time_mean}, Time Std: {time_std}\n",
        )
