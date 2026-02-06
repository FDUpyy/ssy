import numpy as np
import torch
import pywt
from .dwt_multivariate import disentangle, reconstruct


def dwt_revert(input_h, input_l):
    final = []
    for i in range(input_h.shape[0]):

        time = []
        for j in range(input_h.shape[2]):
            coeffs = []
            coeffs_event = disentangle(input_h[i, :, j], "db2", 'symmetric', 2)
            coeffs_trend = disentangle(input_l[i, :, j], "db2", 'symmetric', 2)

            for k in range(len(coeffs_event)):
                coeffs.append(np.array(coeffs_event[j]))
            coeffs[0] = coeffs_trend[0]

            time_series = np.array(reconstruct(coeffs, "db2", 'symmetric')).reshape(-1, 1)
            time.append(time_series)

        time_final = np.concatenate(time, axis=1).reshape(1, -1, input_h.shape[2])
        final.append(time_final)

    out = np.concatenate(final, axis=0)

    return out
