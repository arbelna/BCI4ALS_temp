import numpy as np

"""""
params:
eeg_data - only the data matrix from the channels mXn
marker_ind - a vector with all the indexes of the markers (when each evokes potential begin)
chan_num - the amount of the relevant channels  = to eeg_data -  m
fs  - sampling frequency 
bf_ml - the number of time [ms] that we want before the marker/stimuli
af_ml - the number of time [ms] that we want after the marker/stimuli
ep_ml - the amount of time [ms] that given for the stimulation on the screen

output: 
 segmented_data - a 3D matrix dims order - [segments,channels,eeg_data]
    segment - stimuli number/index
    channel - which channel we are now at
    eeg_data - the actual data from the specific segment and channel
    
"""""


def segmentation(eeg_data, marker_ind, chan_num, fs, bf_ml, af_ml, ep_ml):
    ml_to_ind = 1000 / fs
    before_ep_ind = round(bf_ml / ml_to_ind)  # number of indexes that equal to ~bf_ml[ms]
    after_ep_ind = round(af_ml / ml_to_ind)  # number of indexes that equal to ~af_ml[ms]
    ep_ind_len = round(ep_ml / ml_to_ind)  # number of indexes equal to ~ep_ml[ms]
    segmented_data = np.empty((len(marker_ind), chan_num, before_ep_ind + after_ep_ind + ep_ind_len))

    for seg in range(0, len(marker_ind)):
        segmented_data[seg] = eeg_data[:, marker_ind[seg] - before_ep_ind:marker_ind[seg] + after_ep_ind + ep_ind_len]
        for chan in range(0, chan_num):
            segmented_data[seg, chan] = segmented_data[seg, chan] - np.mean(segmented_data[seg, chan, 0:before_ep_ind])
    return segmented_data
