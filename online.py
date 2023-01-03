from .experiment import Experiment as ex
from .preprocessing_online import PreOnline
from .Prepocessing import Preprocessing
from eeg import Eeg
import numpy as np
import pickle

eeg = Eeg()
exp = ex.Experiment(eeg)

# Start stream
# initialize headset
print("Turning EEG connection ON")
eeg.on()
eeg.clear_board()
exp.run_experiment(eeg)
data = eeg.get_stream_data()
print("Turning EEG connection OFF")
eeg.off()

durations, labels = eeg.extract_trials(data)
labels, durations = np.array(labels)[:, None], np.array(durations)[:, None]
m = exp.results.Block.unique().shape[0]
n = int(labels.shape[0] / exp.results.Block.unique().shape[0])
labels = labels.reshape(m, n)  # Convert array into rows=blocks and columns=trials
durations = durations.reshape(m, n)  # Convert array into rows=blocks and columns=trials
eeg_raw_data = data[1:14]   ### check tomorrow!!!!!!!!!!!!!!!!!!!!!!!!!
fs = eeg.sample_freq
eeg_filter_data = Preprocessing.filter_data(eeg_raw_data.copy(), 1, 40, fs, 50)
eeg_filter_data = eeg_filter_data.copy()/1e06
eeg_data, __ = Preprocessing.ica(eeg_filter_data, fs)

target = np.array(['sad'])
PreOnline.segment(eeg_data, labels, target, durations, fs)
happy, sad, dist = PreOnline.downsampling()
model = pickle.load(open('model123.p', 'rb'))
model.test_model(happy, sad)


