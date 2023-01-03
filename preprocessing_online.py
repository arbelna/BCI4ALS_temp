import numpy as np
import scipy.signal as signal


class PreOnline:
    def __init__(self, target, labels):
        self.target = target
        self.labels = labels
        self.segmented_happy = None
        self.segmented_sad = None
        self.segmentes_distract = None

    def markers(self, target, labels):
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

    def segment(self, eeg_data, labels, target, durations, fs):
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
        dict_idx = self.markers(target, labels)
        target_idx = dict_idx['target_indexes']
        non_target_idx = dict_idx['non_target_indexes']
        dist_idx = dict_idx['distractor_indexes']
        # for blok in range(target.shape[0]):
        segmented_target = np.zeros((eeg_data.shape[0], target_idx.shape[0], fs))
        segmented_non_target = np.zeros((eeg_data.shape[0], non_target_idx.shape[0], fs))
        segmented_dist = np.zeros((eeg_data.shape[0], dist_idx.shape[0], fs))
        seg_tar_del = []
        seg_non_tar_del = []
        seg_dist_del = []
        for channel in range(eeg_data.shape[0]):
            for seg in range(target_idx.shape[0]):
                target = eeg_data[channel][int(durations[0][target_idx[seg]] - 25):int(durations[0][target_idx[seg]] + 100)]
                if eeg_data[channel][int(durations[0][target_idx[seg]] - 25):int(durations[0][target_idx[seg]] + 100)].shape[0] != 125:
                    break
                if np.max(np.abs(target)) < 100e-06:
                    segmented_target[channel][seg] = target
                else:
                    seg_tar_del.append([channel, seg])
            for jj in range(non_target_idx.shape[0]):
                non_target = eeg_data[channel][int(durations[0][non_target_idx[jj]] - 25):int(durations[0][non_target_idx[jj]] + 100)]
                if eeg_data[channel][int(durations[0][non_target_idx[jj]] - 25):int(durations[0][non_target_idx[jj]] + 100)].shape[0] != 125:
                    break
                if np.max(np.abs(non_target)) < 100e-06:
                    segmented_non_target[channel][jj] = non_target
                else:
                    seg_non_tar_del.append([channel, jj])
            for jdx in range(dist_idx.shape[0]):
                distract = eeg_data[channel][int(durations[0][dist_idx[jdx]] - 25):int(durations[0][dist_idx[jdx]] + 100)]
                if eeg_data[channel][int(durations[0][dist_idx[jdx]] - 25):int(durations[0][dist_idx[jdx]] + 100)].shape[0] != 125:
                    break
                if np.max(np.abs(distract)) < 100e-06:
                    segmented_dist[channel][jdx] = distract
                else:
                    seg_dist_del.append([channel, jdx])
        self.segmented_happy = np.delete(segmented_target, seg_tar_del, axis=1)
        self.segmented_sad = np.delete(segmented_non_target, seg_non_tar_del, axis=1)
        self.segmented_dist = np.delete(segmented_dist, seg_dist_del, axis=1)

    def downsampling(self):
        original_sample_rate = 125  # Hz
        new_sample_rate = 82  # Hz
        num_samples = self.segmented_happy.shape[-1]  # 125

        # Compute the original duration of the time series
        original_duration = num_samples / original_sample_rate  # seconds

        # Compute the new duration of the time series
        new_duration = original_duration  # Keep the same duration as the original time series
        new_num_samples = int(new_sample_rate * new_duration)  # Number of samples in the downsampled time series
            # Create an empty array to store the downsampled EEG data
        downsampled_data_happy = np.empty(self.segmented_happy.shape[0], self.segmented_happy.shape[1], new_num_samples)
        downsampled_data_sad = np.empty(self.segmented_sad.shape[0], self.segmented_sad.shape[1], new_num_samples)
        downsampled_data_dist = np.empty(self.segmentes_distract.shape[0], self.segmentes_distract.shape[1], new_num_samples)
        for i in range(self.segmented_happy.shape[0]):
            for j in range(self.segmented_happy.shape[1]):
                time_series = self.segmented_happy[i, j, :]  # Extract the time series
                downsampled_time_series = signal.resample(time_series,new_num_samples)  # Downsample the time series
                downsampled_data_happy[i, j, :] = downsampled_time_series  # Store the downsampled time series
        for i in range(self.segmented_sad.shape[0]):
            for j in range(self.segmented_sad.shape[1]):
                time_series = self.segmented_sad[i, j, :]  # Extract the time series
                downsampled_time_series = signal.resample(time_series,new_num_samples)  # Downsample the time series
                downsampled_data_sad[i, j, :] = downsampled_time_series  # Store the downsampled time series
        for i in range(self.segmentes_distract.shape[0]):
            for j in range(self.segmentes_distract.shape[1]):
                time_series = self.segmentes_distract[i, j, :]  # Extract the time series
                downsampled_time_series = signal.resample(time_series,new_num_samples)  # Downsample the time series
                downsampled_data_dist[i, j, :] = downsampled_time_series  # Store the downsampled time series
        return downsampled_data_happy, downsampled_data_sad, downsampled_data_dist