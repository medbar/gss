import numpy as np
from gss_helpers.stft_utils import linear_resample


def test_linear_resample():
    input_array = np.array([0, 2, 1])
    pred = linear_resample(input_array, 5)
    assert np.allclose(pred, np.array([0, 1, 2, 1.5, 1]))

    pred = linear_resample(input_array, 9)
    assert np.allclose(pred, np.array([0, 0.5, 1, 1.5, 2, 1.75, 1.5, 1.25, 1]))
    pred = linear_resample(input_array, 2)
    assert np.allclose(pred, np.array([0, 1]))
    pred = linear_resample([0, 2, 6, 1], 3)
    assert np.allclose(pred, np.array([0, 4, 1]))
