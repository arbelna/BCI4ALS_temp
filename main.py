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
# import experiment as ex
# from eeg import Eeg
# import numpy as np
import pickle
#
# eeg = Eeg()
# exp = ex.Experiment(eeg)
# eeg.stream_on()
# eeg.clear_board()
# exp.run_experiment(eeg)
# data = eeg.get_stream_data()
# eeg.stream_off()
# with open('ido_raw_data10.npy', 'wb') as f:
#     np.save(f, data, allow_pickle=True)
#
# file = open('ido_record10', 'wb')
# # dump information to that file
# pickle.dump(exp, file)
# # close the file
# file.close()
#
# df = exp.results
# df.to_csv('ido_experiment_results10.csv', index=False)
# print(exp)

import Advaboost as Ab
loc = "downsample_data_tot.npy"
import itertools
from tqdm import tqdm

# check which suns
# channels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
# res_per_comb = []
# for comb in tqdm(range(len(channels) + 1)):
#     for subset in itertools.combinations(channels, comb):
#         if len(subset) < 2:
#             continue
#         temp = Ab.Adva_boost(loc, list(subset))
#         train_score, test_score = temp.train_model()
#         res_per_comb.append([subset, train_score, test_score])
# res_per_comb = res_per_comb.sort(key=lambda x: x[2])

#
temp = Ab.Adva_boost(loc)
temp.train_model()
file = open('model.p', 'wb')
# dump information to that file
pickle.dump(temp, file)

# import model as md
# temp = md.model(loc)
# temp.train_model()
# file = open('model11111.p', 'wb')
# # dump information to that file
# pickle.dump(temp, file)


# temp = pickle.load(open('model.p', 'rb'))
# res = temp.test_model('downsample_test_data.npy')
# print("aaaaaaaaaaaaaaaaaaaaa")
