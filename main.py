import experiment as ex
from eeg import Eeg
import numpy as np
#eeg = Eeg()
exp = ex.Experiment('eeg')
# eeg.stream_on()
# eeg.clear_board()
exp.run_experiment('eeg')
# data = eeg.get_stream_data()
# eeg.stream_off()
# with open('raw_data.npy', 'wb') as f:
#     np.save(f, data, allow_pickle=True)


#df = exp.results
# df.to_csv('experiment_results.csv', index=False)
# print(exp)


