import experiment as ex
from eeg import Eeg
import numpy as np
import pickle

eeg = Eeg()
exp = ex.Experiment(eeg)
eeg.stream_on()
eeg.clear_board()
exp.run_experiment(eeg)
data = eeg.get_stream_data()
eeg.stream_off()
with open('yoav_raw_data1.npy', 'wb') as f:
    np.save(f, data, allow_pickle=True)

file = open('yoav_record(1)', 'wb')
# dump information to that file
pickle.dump(exp, file)
# close the file
file.close()


df = exp.results
df.to_csv('yoav_experiment_results(1).csv', index=False)
print(exp)