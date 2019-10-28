import numpy as np


def padOnes(mat):
    return np.vstack([mat, np.ones((1, mat.shape[1]))])