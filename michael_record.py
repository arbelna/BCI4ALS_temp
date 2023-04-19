import experiment as ex
from eeg import Eeg
import numpy as np
import pickle

num_record = 5
eeg = Eeg()
exp = ex.Experiment(eeg)
eeg.stream_on()
eeg.clear_board()
exp.run_experiment(eeg)
data = eeg.get_stream_data()
eeg.stream_off()
with open(f'michael_exp_{num_record}.npy', 'wb') as f:
    np.save(f, data, allow_pickle=True)
#
file = open(f'michael_exp_{num_record}', 'wb')
# dump information to that file
pickle.dump(exp, file)
# close the file
file.close()

df = exp.results
df.to_csv(f'michael_exp_{num_record}.csv', index=False)
