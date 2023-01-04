from typing import List
import mne
import os
import numpy as np
from nptyping import NDArray
import scipy.signal as signal
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


class Preprocessing:
    def __init__(self, df):
        self.enum_image = {0: 'furious', 1: 'sad', 2: 'happy'}
        self.df = df
        self.dict = {}
        # self.dict = None
        self.tar_arr = None
        self.non_tar_arr = None
        self.dist_arr = None



    def laplacian(self, data: NDArray, channels: List[str]):
        """
        The method execute laplacian on the raw data.
        The laplacian was computed as follows:
            1. C_i = C_i - mean(sum(c_j)) where j is the neighbors of channel i
        The data need to be (n_channel, n_samples)
        Example:
            Electrode; Neighbors
            C3; Cz,FC1,FC5,CP1,CP5
            C4; Cz,FC6,FC2,CP2, CP5
            Cz; C3,C4,FC1,FC2,FC5
            FC1; C3,Cz,FC2,FC5,CP1,CP2,CP5
            FC2; C4,Cz,FC1,FC6,CP1,CP2,CP5
            FC5; C3,Cz,FC1,CP1,CP5
            FC6; FC2,Cz,C4,CP2,CP6
            CP1; FC1,CP2,CP5,C3,FC1
            CP2; FC2,CP1,CP6,c4,FC2
            CP5; FC1,FC5,CP1,c3,O1
            CP6; FC6,FC2,CP2,c4,O2
            O1; CP5,CP1,CP2,O2
            O2; CP6,CP2,CP5,O1

        :return:
        """

        # Dict with all the indices of the channels
        idx = {ch: channels.index(ch) for ch in channels}
        # data[idx['C3']] -= (data[idx['Cz']] + data[idx['FC5']] + data[idx['FC1']] +
        #                     data[idx['CP5']] + data[idx['CP1']]) / 5
        # data[idx['C4']] -= (data[idx['Cz']] + data[idx['FC2']] + data[idx['FC6']] +
        #                     data[idx['CP2']] + data[idx['CP6']]) / 5
        # data[idx['Cz']] -= (data[idx['C3']] + data[idx['C4']] + data[idx['FC1']] + data[idx['FC2']] + data[idx['FC5']]) / 5
        # data[idx['FC1']] -= (data[idx['C3']] + data[idx['Cz']] + data[idx['FC2']] + data[idx['FC5']] + data[idx['CP1']] + data[idx['CP2']] + data[idx['CP5']]) / 7
        # data[idx['FC2']] -= (data[idx['C4']] + data[idx['Cz']] + data[idx['FC1']] + data[idx['FC6']] + data[idx['CP1']] + data[idx['CP2']] + data[idx['CP5']]) / 7
        # data[idx['FC5']] -= (data[idx['C3']] + data[idx['Cz']] + data[idx['FC']] +
        #                     data[idx['CP1']] + data[idx['CP5']]) / 5
        # data[idx['FC6']] -= (data[idx['FC2']] + data[idx['Cz']] + data[idx['C4']] +
        #                     data[idx['CP2']] + data[idx['CP6']]) / 5
        # data[idx['CP1']] -= (data[idx['FC1']] + data[idx['CP2']] + data[idx['CP5']] +
        #                      data[idx['C3']] + data[idx['FC4']]) / 5
        # data[idx['CP2']] -= (data[idx['FC2']] + data[idx['CP1']] + data[idx['CP6']] +
        #                      data[idx['C4']] + data[idx['FC2']]) / 5
        data[idx['CP5']] -= (data[idx['FC1']] + data[idx['CP1']] + data[idx['O1']] +
                             data[idx['C3']] + data[idx['FC2']]) / 5
        data[idx['CP6']] -= (data[idx['FC6']] + data[idx['CP2']] + data[idx['O2']] +
                             data[idx['C4']] + data[idx['FC2']]) / 5
        data[idx['O1']] -= (data[idx['CP5']] + data[idx['CP1']] + data[idx['O2']] +
                             data[idx['CP2']]) / 4
        data[idx['O1']] -= (data[idx['CP6']] + data[idx['CP2']] + data[idx['O1']] +
                             data[idx['CP5']]) / 4
        # return data[[idx['C3'], idx['C4']]]
        return data

    def filter_data(self, data, lowcut, highcut, sample_freq, notch_freq):
        y = mne.filter.notch_filter(data, Fs=sample_freq, freqs=notch_freq, verbose=False)
        d = mne.filter.filter_data(y, sample_freq, lowcut, highcut, verbose=False)

        return d

    def markers_dict(self, target, labels):
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

    def segmentation(self, eeg_data, labels, tar, durations):
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
        dict_idx = self.markers_dict(tar, labels)
        target_idx = dict_idx['target_indexes']
        non_target_idx = dict_idx['non_target_indexes']
        dist_idx = dict_idx['distractor_indexes']
        for blok in range(durations.shape[0]):
            self.dict[f'Block_{tar[blok]}'] = {}
            segmented_target = np.empty(eeg_data.shape[0], dtype='object')
            segmented_non_target = np.empty(eeg_data.shape[0], dtype='object')
            segmented_dist = np.empty(eeg_data.shape[0], dtype='object')
            for channel in range(eeg_data.shape[0]):
                segmented_target_ch = []
                segmented_non_target_ch = []
                segmented_dist_ch = []
                for seg in range(target_idx[blok].shape[0]):
                    target = eeg_data[channel][int(durations[blok][target_idx[blok][seg]] - 25):int(
                        durations[blok][target_idx[blok][seg]] + 100)]
                    if eeg_data[channel][int(durations[blok][target_idx[blok][seg]] - 25):int(
                            durations[blok][target_idx[blok][seg]] + 100)].shape[0] != 125:
                        break
                    if np.max(np.abs(target)) < 100e-06:
                        segmented_target_ch.append(target)
                segmented_target_ch = np.array(segmented_target_ch, dtype='object')
                for jj in range(non_target_idx[blok].shape[0]):
                    non_target = eeg_data[channel][int(durations[blok][non_target_idx[blok][jj]] - 25):int(
                        durations[blok][non_target_idx[blok][jj]] + 100)]
                    if eeg_data[channel][int(durations[blok][non_target_idx[blok][jj]] - 25):int(
                            durations[blok][non_target_idx[blok][jj]] + 100)].shape[0] != 125:
                        break
                    if np.max(np.abs(non_target)) < 100e-06:
                        segmented_non_target_ch.append(non_target)
                segmented_non_target_ch = np.array(segmented_non_target_ch, dtype='object')
                for jdx in range(dist_idx[blok].shape[0]):
                    distract = eeg_data[channel][int(durations[blok][dist_idx[blok][jdx]] - 25):int(
                        durations[blok][dist_idx[blok][jdx]] + 100)]
                    if eeg_data[channel][int(durations[blok][dist_idx[blok][jdx]] - 25):int(
                            durations[blok][dist_idx[blok][jdx]] + 100)].shape[0] != 125:
                        break
                    if np.max(np.abs(distract)) < 100e-06:
                        segmented_dist_ch.append(distract)
                segmented_dist_ch = np.array(segmented_dist_ch, dtype='object')
                segmented_target[channel] = segmented_target_ch
                segmented_non_target[channel] = segmented_non_target_ch
                segmented_dist[channel] = segmented_dist_ch
            self.dict[f'Block_{tar[blok]}']['target'] = segmented_target
            self.dict[f'Block_{tar[blok]}']['non target'] = segmented_non_target
            self.dict[f'Block_{tar[blok]}']['distractor'] = segmented_dist
        # self.dict


    def mean_trials(self, arrays_of_blocks):
        """
        This function gets the segmented data array and returns the mean of
        all the trials within each block subtracted with the mean of the
         starting gap we took from each marker
         - segmented_data: an array of shape (No. of blocks, No. of channels, No. of samples)
         - fs: sampling frequency
        """
        segmented_data_target_mean = []
        segmented_data_target_var = []
        segmented_data_non_target_mean = []
        segmented_data_non_target_var = []
        segmented_data_dist_mean = []
        segmented_data_dist_var = []
        for blok in range(arrays_of_blocks.shape[0]):
            for type in range(1, arrays_of_blocks[blok].shape[0]):
                for chan in range(arrays_of_blocks[blok, type].shape[0]):
                    irelevant_mean = np.array(
                        np.mean(arrays_of_blocks[blok, type][chan][:, :25], axis=1, keepdims=True))
                    sub_irelevant_mean = arrays_of_blocks[blok, type][
                                             chan] - irelevant_mean  # subtraction of mean before the event from the data
                    if type == 1:
                        segmented_data_target_mean.append(np.array(np.mean(sub_irelevant_mean[:, :], axis=0)))
                        segmented_data_target_var.append(
                            np.array(np.std(sub_irelevant_mean[:, :] / np.sqrt(sub_irelevant_mean.shape[1]))))
                    elif type == 2:
                        segmented_data_non_target_mean.append(np.array(np.mean(sub_irelevant_mean[:, :], axis=0)))
                        segmented_data_non_target_var.append(
                            np.array(np.std(sub_irelevant_mean[:, :] / np.sqrt(sub_irelevant_mean.shape[1]))))
                    else:
                        segmented_data_dist_mean.append(np.array(np.mean(sub_irelevant_mean[:, :], axis=0)))
                        segmented_data_dist_var.append(
                            np.array(np.std(sub_irelevant_mean[:, :] / np.sqrt(sub_irelevant_mean.shape[1]))))
        segmented_data_target_mean = np.array(segmented_data_target_mean)
        segmented_data_target_var = np.array(segmented_data_target_var)
        segmented_data_non_target_mean = np.array(segmented_data_non_target_mean)
        segmented_data_non_target_var = np.array(segmented_data_non_target_var)
        segmented_data_dist_mean = np.array(segmented_data_dist_mean)
        segmented_data_dist_var = np.array(segmented_data_dist_var)
        return segmented_data_target_mean, segmented_data_non_target_mean, segmented_data_dist_mean, segmented_data_target_var, segmented_data_non_target_var, segmented_data_dist_var


    def ica(self, eeg_filter_data, fs):
        ch_types = ['eeg'] * len(eeg_filter_data)
        ch_names = ["C3", "C4", "Cz", "FC1", "FC2", "FC5", "FC6", "CP1", "CP2", "CP5", "CP6", "O1", "O2"]
        info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types=ch_types)
        raw = mne.io.RawArray(eeg_filter_data, info, verbose=False)
        ica_ = mne.preprocessing.ICA(n_components=13, max_iter='auto', random_state=97)
        ten_twenty_montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(ten_twenty_montage)
        ica_.fit(raw)
        ica_.plot_sources(raw, show_scrollbars=False)
        ica_.plot_components()
        ica_.exclude = [10]
        reconst_raw = raw.copy()
        raw_clean = ica_.apply(reconst_raw)
        ica_.plot_sources(raw_clean, show_scrollbars=False)
        data, times = raw_clean[:, :]
        return data, times


    def downsampling(self, online):
        blocks_num = list(self.dict.keys())
        types_list = list(self.dict[blocks_num[0]].keys())
        # Select the sample rate and number of samples
        original_sample_rate = 125  # Hz
        new_sample_rate = 82  # Hz
        num_samples = self.dict[blocks_num[0]][types_list[0]][0].shape[-1]  # 125

        # Compute the original duration of the time series
        original_duration = num_samples / original_sample_rate  # seconds

        # Compute the new duration of the time series
        new_duration = original_duration  # Keep the same duration as the original time series
        new_num_samples = int(new_sample_rate * new_duration)  # Number of samples in the downsampled time series
        downsample_dict = {}
        for idx in blocks_num:  # Go over all the blocks
            downsample_dict[idx] = {}
            for jj in types_list:  # going over channels
                downsampled = np.empty(self.dict[idx][jj].shape[0], dtype='object')
                for chan in range(self.dict[idx][jj].shape[0]):
                    downsample = []
                    for i in range(self.dict[idx][jj][chan].shape[0]):  ## going over trials
                        time_series = self.dict[idx][jj][chan][i, :]  # Extract the time series
                        downsampled_time_series = signal.resample(time_series,new_num_samples)  # Downsample the time series
                        downsample.append(downsampled_time_series)
                    downsample = np.array(downsample, dtype='object')
                    downsampled[chan] = downsample
                downsample_dict[idx][jj] = downsampled
            if online:
                self.tar_arr = downsample_dict['Block_sad']['target']
                self.non_tar_arr = downsample_dict['Block_sad']['non target']
                self.dist_arr = downsample_dict['Block_sad']['distractor']
            else:
                self.tar_arr = np.concatenate((downsample_dict['Block_happy']['target'], downsample_dict['Block_sad']['target']))
                self.non_tar_arr = np.concatenate((downsample_dict['Block_happy']['non target'], downsample_dict['Block_sad']['non target']))
                self.dist_arr = np.concatenate((downsample_dict['Block_happy']['distractor'], downsample_dict['Block_sad']['distractor']))
