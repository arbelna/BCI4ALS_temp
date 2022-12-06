import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
# data = pd.read_table('OpenBCI-RAW-2022-11-24_15-56-06.txt', sep=", ", header=4, engine='python')
# x = np.arange(1, len(data.loc[:, "EXG Channel 9"])+1, 1)
# y = data.loc[:, "EXG Channel 9"]
# plt.plot(x, y)
# plt.show()
# Program to calculate moving average using numpy

# def smooth_mvg( arr -> NDArray,window_size -> int)
import numpy as np
from matplotlib import pyplot as plt
import mne


def moving_avg(arr, window_size):
    for j in range(arr.shape[0]):
        moving_averages = [arr[j, 0]]
        i = 0
        while i < arr.shape[1] - window_size + 1:
            window_average = np.sum(arr[j, i:i + window_size]) / window_size
                       # window in moving average list
            moving_averages.append(window_average)
            # Shift window to right by one position
            i += 1
        if j == 0:
            new_arr = np.zeros((arr.shape[0], len(moving_averages)-1))
        new_arr[j, :] = moving_averages[1:]
    return new_arr


# Initialize an empty list to store moving averages
# Loop through the array t o
# consider every window of size 3

data_raw_all = np.load("../records/Itamar1.npy")
win_sz = 100
data_mv_avg = moving_avg(data_raw_all[1:14, :], win_sz)
data_notch_filll = mne.filter.notch_filter(data_mv_avg, 125, 50)
data_band_notch = mne.filter.filter_data(data_notch_filll, 125, 0.5, 40)



plt.plot(data_mv_avg[3, :])
plt.show()

# ctrl +alt +l - mesader akol
# ctrl +d   - duplicate
