from torch import nn
import torch
import numpy as np
from abc import ABC
from module.utils import weight_init, top_n_accuracy
from module.subutils import seq2seq_forward,rnn_forward


class MCLocPredictor:
    def __init__(self, num_loc):
        self.transfer_mat = np.zeros((num_loc, num_loc))

    def fit(self, sequences):
        """
        @param sequences: sequences for training.
        """
        for sequence in sequences:
            for s, e in zip(sequence[:-1], sequence[1:]):
                self.transfer_mat[s, e] += 1

    def predict(self, src_seq, pre_len):
        pre_seq = []
        s = src_seq[-pre_len-1]
        for i in range(pre_len):
            pre = np.argmax(self.transfer_mat[s])
            pre_seq.append(pre)
            s = pre
        return pre_seq