import numpy as np
import torch
import pywt
def disentangle(data, wavelet, mode, j):
    # temporal_to_wavelet
    #data = data
    coeffs = pywt.wavedec(data, wavelet, mode, j)
    return coeffs

def reconstruct(data, wavelet, mode):
    # coef_to_temporal
    time_series = pywt.waverec(data, wavelet, mode)
    return time_series

#没用上：
def disentangle_multi(data, wavelet, mode, j):
    # temporal_to_wavelet
    #data = data
    coeffs = pywt.wavedec(data, wavelet, mode, j)
    return coeffs[0],coeffs[1],coeffs[2]

def dwt_multivariate(data_multivariate):
    # temporal_to_wavelet
    event = []
    trend = []
    # i = data_multivariate.shape[1]
    for i in range(data_multivariate.shape[1]):

        coeffs = disentangle(data_multivariate[:, i], "db2", 'symmetric', 2)
        coeffs_trend = []
        coeffs_event = []

        # coef_extraction
        for j in range(len(coeffs)):
            coeffs_event.append(np.array(coeffs[j]))
            coeffs_trend.append(np.array(coeffs[j]))
            coeffs_trend[j] = np.zeros_like(coeffs_trend[j])
        coeffs_event[0] = np.zeros_like(coeffs_event[0])  # 将低频系数赋0，只保留其余高频系数
        coeffs_trend[0] = np.array(coeffs[0])  # 只保留唯一的低频系数，其余多级高频系数全赋0

        # high_coef_to_temporal
        time_series_event = np.array(reconstruct(coeffs_event, "db2", 'symmetric')).reshape(-1, 1)
        # time_series_event = (number 1)
        event.append(time_series_event)

        # low_coef_to_temporal
        time_series_trend = np.array(reconstruct(coeffs_trend, "db2", 'symmetric')).reshape(-1, 1)
        trend.append(time_series_trend)
        """
        # figure parameter
        index = np.arange(len(data_multivariate))
        plt.figure(i, figsize=(10, 6))

        # draw original temporal figure
        plt.subplot(3, 1, 1)
        plt.plot(index, data_multivariate[:, i], label='original data')
        plt.legend(loc='upper right')

        # draw event figure
        plt.subplot(3, 1, 2)
        plt.plot(index, time_series_event, label='event_high_frequency')
        plt.legend(loc='upper right')

        # draw trend figure
        plt.subplot(3, 1, 3)
        plt.plot(index, time_series_trend, label='trend_low_frequency')
        plt.legend(loc='upper right')

        plt.show()
        plt.clf()
        """
    event_final = np.concatenate(event, axis=1)
    trend_final = np.concatenate(trend, axis=1)
    final = np.concatenate((event_final, trend_final), axis=1)

    return final
    # print(event_final.shape)
    # print(trend_final.shape)