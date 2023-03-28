import numpy as np


def linear_resample(y, n):
    """
    Args:
        y: Input Array . shape = (seq_len)
        n: Len of the output Array

    Returns: np.array with shape of (n)
    """
    x = np.linspace(0, n, len(y))
    target_x = np.linspace(0, n, n)
    y_interp = np.interp(target_x, x, y)
    return y_interp
