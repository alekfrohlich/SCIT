"""Implementation of Nonstandard Distributions."""

import numpy as np
from numpy import ndarray


def checkerboard(sample_size: int, bins: int, d: int) -> ndarray:
    """Checkerboard distribution over [0,1)^d."""
    xy = np.zeros((sample_size, d))

    # Rejection-based sampling
    for i in range(sample_size):
        rejection_criterion = True

        xy_candidate = None
        while rejection_criterion:
            xy_candidate = np.random.rand(1, d)
            rejection_criterion = (np.floor(bins * xy_candidate).sum() % 2) == 1

        xy[i] = xy_candidate

    return xy[:, : d // 2], xy[:, d // 2 :]
