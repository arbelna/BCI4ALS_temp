from matplotlib import pyplot as plt
import mne
import numpy as np
from moving_avg import moving_avg
from segment import segmentation
from ans_avg import ans_avg

# Initialize an empty list to store moving averages
# Loop through the array t o
# consider every window of size 3

data_raw_all = np.load("../records/Itamar1.npy")
win_sz = 100
data_mv_avg = moving_avg(data_raw_all[1:14, :], win_sz)
data_notch_filll = mne.filter.notch_filter(data_mv_avg, 125, 50)
data_band_notch = mne.filter.filter_data(data_notch_filll, 125, 0.5, 40)

seg_dat = segmentation(data_band_notch, np.array([500, 2500, 3500, 4500, 5500, 6500]), 13, 125, 200, 500, 300)

avg_cha = np.zeros((13, 125))
"""
in this loop we do ensemble averaging on all the eeg channels  
and using the function ans_avg in every iteration
"""
for i in range(0, 13):
    avg_cha[i, :] = np.transpose(ans_avg(seg_dat[:, i, :], 6))

print(avg_cha)

# ctrl +alt +l - mesader akol
# ctrl +d   - duplicate
