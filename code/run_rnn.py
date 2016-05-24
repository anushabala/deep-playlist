__author__ = 'kalpit'

import tensorflow as tf
from get_data import *

if __name__ == '__main__':
    ##### GET DATA #####
    X, y, word_counts = get_data('../lastfm_train_mappings.txt', '../lyrics/data/lyrics/train/', threshold=100)

    keys, vals = word_counts.keys(), word_counts.values()
    keys, vals = [list(x) for x in zip(*sorted(zip(keys, vals), key=lambda x: x[1], reverse=True))]

    print len(keys)
    ##### RUN RNN #####
