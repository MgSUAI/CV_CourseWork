
import numpy as np
from scipy.spatial import distance


def dist_manhattan(F1, F2):
    """
    Compute the Manhattan distance between two feature descriptors.
    """
    diff = np.abs(F1 - F2)
    mdist = np.sum(diff)
    return mdist


def dist_euclidean(F1, F2):
    """
    Compare two feature descriptors using Euclidean distance.
    """
    diff = F1 - F2
    dst = np.sqrt(np.sum(diff * diff))
    return dst


def dist_mahalanobis(F1, F2, cov_inv):
    """
    Compute the Mahalanobis distance between two feature descriptors.
    """
    diff = F1 - F2
    mdist_squared = diff.T @ cov_inv @ diff
    mdist = np.sqrt(mdist_squared)

    # mdist = distance.mahalanobis(F1, F2, cov_inv)
    return mdist


def dist_chi_squared(F1, F2):
    """
    Compute the Chi-squared distance between two feature descriptors.
    """
    eps = 1e-10  # Small value to avoid division by zero
    chi_sq = 0.5 * np.sum(((F1 - F2) ** 2) / (F1 + F2 + eps))
    return chi_sq

