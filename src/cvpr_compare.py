
import numpy as np

def cvpr_compare(F1, F2):
    """
    Compare two feature descriptors using Euclidean distance.
    """
    diff = F1 - F2
    dst = np.sqrt(np.sum(diff * diff))
    return dst
