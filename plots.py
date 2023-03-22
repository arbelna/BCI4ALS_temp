import matplotlib.pyplot as plt
import numpy as np
import os

class Plots:
    def __init__(self, fs, eeg_channels, distractor, path, dir_name, time):
        self.fs = fs
        self.time = np.linspace(-200, 800, fs)
        self.eeg_ch = eeg_channels
        self.distractor = distractor  # True of False, decide if to plot the disractor or not
        self.records_path = path  # Path to plots directory
        self.dir_name = dir_name  # Name of the folder we want to create
        self.time = time  # True or False, decide if to plot x axis as time or samples

    def p300_plots(self, segmented_data_target_mean, segmented_data_non_target_mean, segmented_data_non_target_std, segmented_data_target_std, segmented_data_dist_mean):
        os.mkdir(f'{self.records_path}/{self.dir_name}')
        os.mkdir(f'{self.records_path}/{self.dir_name}/all')
        os.mkdir(f'{self.records_path}/{self.dir_name}/tar vs non_tar')
        for block in range(segmented_data_target_mean.shape[0]):
            for chan in range(segmented_data_non_target_mean.shape[1]):
                if self.time:
                    plt.figure(1, figsize=(18, 5))
                    plt.subplot(1, 3, 1)
                    plt.plot(self.time, segmented_data_non_target_mean[block][chan], 'b', label='non target')
                    plt.fill_between(self.time, segmented_data_non_target_mean[block][chan] - segmented_data_non_target_std[block][chan],
                                     segmented_data_non_target_mean[block][chan] + segmented_data_non_target_std[block][chan],
                                     color='b', alpha=0.2)
                    plt.title(f'Block No.: {block}, channel {self.eeg_ch[chan]}; with std')
                    plt.xlabel('Time [ms]')
                    plt.ylabel('Power [v]')
                    plt.grid()
                    plt.legend(loc='best')
                    plt.subplot(1, 3, 2)
                    plt.plot(self.time, segmented_data_target_mean[block][chan], 'g', label='target')
                    plt.fill_between(self.time, segmented_data_target_mean[block][chan] - segmented_data_target_std[block][chan],
                                     segmented_data_target_mean[block][chan] + segmented_data_target_std[block][chan], color='g',
                                     alpha=0.1)
                    plt.title(f'Block No.: {block}, channel {self.eeg_ch[chan]}; with std')
                    plt.xlabel('Time [ms]')
                    plt.ylabel('Power [v]')
                    plt.grid()
                    plt.legend(loc='best')
                    plt.subplot(1, 3, 3)
                    plt.plot(self.time, segmented_data_non_target_mean[block][chan], 'b', label='non target')
                    plt.plot(self.time, segmented_data_target_mean[block][chan], 'g', label='target')
                    if self.distractor == True:
                        plt.plot(self.time, segmented_data_dist_mean[block][chan], label='distractor')
                    plt.title(f'Block No.: {block}, channel {self.eeg_ch[chan]}')
                    plt.xlabel('Time [ms]')
                    plt.ylabel('Power [v]')
                    plt.grid()
                    plt.legend(loc='best')
                    if self.distractor == True:
                        plt.savefig(f'{self.records_path}/{self.dir_name}/all/block_{block}_channel_{chan}.png', bbox_inches='tight')
                    elif self.distractor == False:
                        plt.savefig(f'{self.records_path}/{self.dir_name}/tar vs non_tar/block_{block}_channel_{chan}.png',bbox_inches='tight')
                else:
                    plt.figure(1, figsize=(18, 5))
                    plt.subplot(1, 3, 1)
                    plt.plot(segmented_data_non_target_mean[block][chan], 'b', label='non target')
                    plt.fill_between(
                                     segmented_data_non_target_mean[block][chan] - segmented_data_non_target_std[block][
                                         chan],
                                     segmented_data_non_target_mean[block][chan] + segmented_data_non_target_std[block][
                                         chan],
                                     color='b', alpha=0.2)
                    plt.title(f'Block No.: {block}, channel {self.eeg_ch[chan]}; with std')
                    plt.xlabel('Samples')
                    plt.ylabel('Power [v]')
                    plt.grid()
                    plt.legend(loc='best')
                    plt.subplot(1, 3, 2)
                    plt.plot(segmented_data_target_mean[block][chan], 'g', label='target')
                    plt.fill_between(
                                     segmented_data_target_mean[block][chan] - segmented_data_target_std[block][chan],
                                     segmented_data_target_mean[block][chan] + segmented_data_target_std[block][chan],
                                     color='g',
                                     alpha=0.1)
                    plt.title(f'Block No.: {block}, channel {self.eeg_ch[chan]}; with std')
                    plt.xlabel('Samples')
                    plt.ylabel('Power [v]')
                    plt.grid()
                    plt.legend(loc='best')
                    plt.subplot(1, 3, 3)
                    plt.plot(segmented_data_non_target_mean[block][chan], 'b', label='non target')
                    plt.plot(segmented_data_target_mean[block][chan], 'g', label='target')
                    if self.distractor:
                        plt.plot(segmented_data_dist_mean[block][chan], label='distractor')
                    plt.title(f'Block No.: {block}, channel {self.eeg_ch[chan]}')
                    plt.xlabel('Samples')
                    plt.ylabel('Power [v]')
                    plt.grid()
                    plt.legend(loc='best')
                    if self.distractor:
                        plt.savefig(f'{self.records_path}/{self.dir_name}/all/block_{block}_channel_{chan}.png',
                                    bbox_inches='tight')
                    elif self.distractor == False:
                        plt.savefig(
                            f'{self.records_path}/{self.dir_name}/tar vs non_tar/block_{block}_channel_{chan}.png',
                            bbox_inches='tight')

                plt.show()