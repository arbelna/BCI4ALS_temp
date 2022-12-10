import numpy as np


def ans_avg(sig_mat, stim_num):
    ans_avg_sig = (1 / stim_num) * np.matmul(np.transpose(sig_mat), np.ones((stim_num, 1)))
    return ans_avg_sig
