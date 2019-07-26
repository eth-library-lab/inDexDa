import numpy as np


def train_test_sets(model, options):
    split = 0.8
    train_positives = int(options.num_positives * split)
    train_negatives = int(options.num_negatives * split)
    test_positives = int(options.num_positives * split)
    test_negatives = int(options.num_negatives * split)

    train_samples = train_positives + train_negatives
    test_samples = test_positives + test_negatives

    train_arrays = np.zeros((train_samples, options.vocab_size))
    train_labels = np.zeros(train_samples)

    for i in range(train_positives):
        prefix_train_pos = 'TRAIN_POS_' + str(i)
        train_arrays[i] = model[prefix_train_pos]
        train_labels[i] = 1

    for j in range(train_negatives):
        prefix_train_neg = 'TRAIN_NEG_' + str(i)
        train_arrays[train_positives + i] = model[prefix_train_neg]
        train_labels[train_positives + i] = 0

    test_arrays = np.zeros((test_samples, options.vocab_size))
    test_labels = np.zeros(test_samples)

    for i in range(test_positives):
        prefix_test_pos = 'TEST_POS_' + str(i)
        test_arrays[i] = model[prefix_test_pos]
        test_labels[i] = 1

    for i in range(test_negatives):
        prefix_test_neg = 'TEST_NEG_' + str(i)
        test_arrays[test_positives + i] = model[prefix_test_neg]
        test_labels[test_positives + i] = 0

    train = Data(train_arrays, train_labels)
    test = Data(test_arrays, test_labels)
    return train, test


class Data():
    def __init__(self, arrays, labels):
        self.arrays = arrays
        self.labels = labels
