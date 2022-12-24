import numpy as np
import pandas as pd
import os
import mne
import matplotlib.pyplot as plt
from BCI4ALS.eeg import Eeg

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
records_path = f'{__location__}/BCI4ALS/records/Ido - happy, sad'
with open(f'{records_path}/ido_raw_data.npy', 'rb') as f:
    raw_data = np.load(f)

raw_data_df = pd.read_csv(f'{records_path}/ido_experiment_results.csv')


# eeg = Eeg
def decode_marker(marker_value):
    """
    Decode the marker and return a tuple with the status, label and index.
    Look for the encoder docs for explanation for each argument in the marker.
    :param marker_value:
    :return:
    """
    if marker_value % 10 == 1:
        status = "start"
        marker_value -= 1
    elif marker_value % 10 == 2:
        status = "stop"
        marker_value -= 2
    else:
        raise ValueError("incorrect status value. Use start or stop.")

    label = ((marker_value % 100) - (marker_value % 10)) / 10

    index = (marker_value - (marker_value % 100)) / 100

    return status, int(label), int(index)


def extract_trials(data):
    """
    The method get ndarray and extract the labels and durations from the data.
    :param data: the data from the board.
    :return:
    """
    marker_row = 31
    # Init params
    durations, labels = [], []

    # Get marker indices
    markers_idx = np.where(data[marker_row, :] != 0)[0]

    # For each marker
    for idx in markers_idx:

        # Decode the marker
        status, label, _ = decode_marker(data[marker_row, idx])

        if status == 'start':

            labels.append(label)
            durations.append((idx,))

        elif status == 'stop':

            durations[-1] += (idx,)

    return durations, labels


def filter_data(data, lowcut, highcut, sample_freq, notch_freq):
    y = mne.filter.notch_filter(data, Fs=sample_freq, freqs=notch_freq, verbose=False)
    d = mne.filter.filter_data(y, sample_freq, lowcut, highcut, verbose=False)
    return d

def markers_dict(target, labels):
    dict_idx = {'distractor_indexes': [], 'target_indexes': [], 'non_target_indexes': []}
    for idx in range(target.shape[0]):
        dict_idx['distractor_indexes'].append(np.where(labels[idx] == 0)[0])
        if target[idx] == 'sad':
            dict_idx['target_indexes'].append(np.where(labels[idx] == 1)[0])
            dict_idx['non_target_indexes'].append(np.where(labels[idx] == 2)[0])
        elif target[idx] == 'happy':
            dict_idx['target_indexes'].append(np.where(labels[idx] == 2)[0])
            dict_idx['non_target_indexes'].append(np.where(labels[idx] == 1)[0])
    return dict_idx

def segmentation(eeg_data, labels, target, durations, fs):
    """
    This function segment the filtered eeg data to trials where each segment
    is a trial within the block
    dict_idx: a dictionary with 3 keys, where each key is the indexes of each type
                (e.g. distractor, target, non-target) and each value is an array of arrays
                with shape of (No. of blocks, No. of the indexes of this type)
    input:
    - eeg_data: eeg_data after filters (for now without pca)
    - labels: Array which contains the label of each trial
    - target: An array with target value at each block
    - durations: An array of marker indexes with shape of (No. of blocks, No. of trials)
    - fs: sample frequency 125
    """
    dict_idx = markers_dict(target, labels)
    # dict_keys_arr = np.array(list(dict_idx.keys()))
    target_idx = dict_idx['target_indexes'][0]
    non_target_idx = dict_idx['non_target_indexes'][0]
    dist_idx = dict_idx['distractor_indexes'][0]
    segmented_data_target = np.zeros((eeg_data.shape[0], durations.shape[0], target_idx.shape[0], fs))
    segmented_data_non_target = np.zeros((eeg_data.shape[0], durations.shape[0], non_target_idx.shape[0], fs))
    segmented_dist_data = np.zeros((eeg_data.shape[0], durations.shape[0], dist_idx.shape[0], fs))
    for channel in range(eeg_data.shape[0]):
        for i in range(durations.shape[0]):
            for seg in range(target_idx.shape[0]):
                if eeg_data[channel][int(durations[i][target_idx[seg]] - 25):int(durations[i][target_idx[seg]] + 100)].shape[0] != 125:
                    break
                segmented_data_target[channel][i][seg] = eeg_data[channel][
                                                         int(durations[i][target_idx[seg]] - 25):int(
                                                             durations[i][target_idx[seg]] + 100)]
            for jj in range(non_target_idx.shape[0]):
                if eeg_data[channel][int(durations[i][non_target_idx[jj]] - 25):int(durations[i][non_target_idx[jj]] + 100)].shape[0] != 125:
                    break
                segmented_data_non_target[channel][i][jj] = eeg_filter_data[channel][
                                                        int(durations[i][non_target_idx[jj]] - 25):int(
                                                            durations[i][non_target_idx[jj]] + 100)]
            for jdx in range(dist_idx.shape[0]):
                if eeg_data[channel][int(durations[i][dist_idx[jdx]] - 25):int(durations[i][dist_idx[jdx]] + 100)].shape[0] != 125:
                    break
                segmented_dist_data[channel][i][jdx] = eeg_filter_data[channel][
                                                            int(durations[i][dist_idx[jdx]] - 25):int(
                                                                durations[i][dist_idx[jdx]] + 100)]
    return segmented_data_target, segmented_data_non_target, segmented_dist_data

def mean_trials(segmented_data, fs):
    """
    This function gets the segmented data array and returns the mean of
    all the trials within each block subtracted with the mean of the
     starting gap we took from each marker
     - segmented_data: an array of shape (No. of channels, No. of blocks, No. of markers for each type, No. of samples)
     - fs: sampling frequency
    """
    irelevant_mean = np.mean(segmented_data[:, :, :25, :], axis=2, keepdims=True)
    irelevant_mean = irelevant_mean.reshape(segmented_data.shape[0], segmented_data.shape[1], fs)
    relevant_mean = np.mean(segmented_data, axis=2, keepdims=True)
    relevant_mean = relevant_mean.reshape(segmented_data.shape[0], segmented_data.shape[1], fs)
    data_mean = relevant_mean - irelevant_mean
    return data_mean

# def run():
    # eeg = Eeg()
    # exp = ex.Experiment(eeg)
    # self.subject_directory = self._ask_subject_directory()
    # self.session_directory = self.create_session_folder(self.subject_directory)

    # Start stream
    # initialize headset
    # print("Turning EEG connection ON")
    # self.eeg.on()
    # self.eeg.clear_board()
    # exp.run_experiment(eeg)
    # data = eeg.get_stream_data()
    # print("Turning EEG connection OFF")
    # self.eeg.off()
    # self.offline.export_files(data)


durations, labels = extract_trials(raw_data)  # channel 30 is the timestamp channel
labels, durations = np.array(labels)[:, None], np.array(durations)[:, None]
m = raw_data_df.Block.unique().shape[0]
n = int(labels.shape[0] / raw_data_df.Block.unique().shape[0])
labels = labels.reshape(m, n)  # Convert array into rows=blocks and columns=trials
durations = durations.reshape(m, n)  # Convert array into rows=blocks and columns=trials
eeg_raw_data = raw_data[1:14]
fs = 125
eeg_filter_data = filter_data(eeg_raw_data.copy(), 1, 40, fs, 50)
target = np.array(['happy', 'sad'])
segmented_data_target, segmented_data_non_target, segmented_dist_data = segmentation(eeg_filter_data, labels, target, durations, fs)
segmented_data_target_mean = mean_trials(segmented_data_target, fs)
segmented_data_non_target_mean = mean_trials(segmented_data_non_target, fs)
segmented_dist_data_mean = mean_trials(segmented_dist_data, fs)

time = np.linspace(-0.2, 0.8, fs)
for block in range(raw_data_df.Block.unique().shape[0]):
    for i in range(segmented_data_non_target_mean.shape[0]):
        plt.figure()
        plt.plot(time, segmented_data_non_target_mean[i][block], '--s', label='non target')
        plt.plot(time, segmented_data_target_mean[i][block], '-s', label='target')
        plt.title(f'Block No.: {block}, channel No. {i}')
        plt.xlabel('Time')
        plt.ylabel('Power')
        plt.legend(loc='best')
        plt.savefig(f'{records_path}/plots/block_{block}_channel_{i}.pdf', bbox_inches='tight')
        plt.show()

# eeg_filter_data = eeg_filter_data.copy() / 1000000
# n_components = 3
# ch_types = ['eeg'] * len(eeg_filter_data)
# ch_names = ["C3", "C4", "Cz", "FC1", "FC2", "FC5", "FC6", "CP1", "CP2", "CP5", "CP6", "O1", "O2"]
# info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types=ch_types)
# raw = mne.io.RawArray(eeg_filter_data, info, verbose=False)
# ica = mne.preprocessing.ICA(n_components=n_components, random_state=42)
# ica.fit(raw)

