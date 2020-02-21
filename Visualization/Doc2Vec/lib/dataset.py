import os
import json
import random
import torch.utils.data as data

from Doc2Vec.lib.normalize_text import Normalize


class Dataset(data.Dataset):
    def __init__(self, train=True, datapath=None):
        '''
        Reads dataset from a json file. Normalizes and tokenizes the dataset as well as
         removes uncommon words (to better handle them in machine learning applications).

        :params  training_data(list of lists)     : samples for training
                 testing_data(list of lists)      : samples for testing
                 min_thresh(int)                  : minimum word count threshold
                 max_thresh(int)                  : maximum word count threshold
                 train_test_split(float)          : train test split
                 train(bool)                      : using training data
        '''
        self.train = train
        self.data = self.read_dataset(datapath)

    def read_dataset(self, datapath):
        with open(datapath, 'r') as f:
            contents = f.read()
            datasets = json.loads(contents)

            normalized_datasets = []
            for name in datasets:
                normalized_keywords = []
                for keyword in datasets[name]:
                    normalize = Normalize(keyword)
                    normalized_keywords.extend(normalize.normalized_text)
                normalized_datasets.append([name, normalized_keywords])

        return normalized_datasets

    def __getitem__(self, index):
            return self.data[index][0], self.data[index][1]

    def __len__(self):
        return len(self.data)
