import os
import json
import random
import inspect
import numpy as np
import torch.utils.data as data


class Dataset(data.Dataset):
    def __init__(self, train=True, min_thresh=10, max_thresh=None,
                 train_test_split=0.8):
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
        self.train_test_split = train_test_split
        self.all_samples = self.get_dataset()

        self.partition_dataset(self.all_samples)

    def get_dataset(self):
        dataset = []
        with open('/home/parker/code/datadex/NaturalLanguageProcessing/LSTM/log/dataset.json', 'r') as f:
            contents = f.read()
            try:
                raw = json.loads(contents)
                for paper in raw:
                    dataset.append([paper['Text'], paper['Target']])
            except ValueError:
                print('Not able to parse json file to dictionary.\n')

        return dataset

    def partition_dataset(self, dataset):
        # Split dataset into testing and training samples
        random.seed(100)
        random.shuffle(dataset)

        if self.train:
            self.data = dataset[:int(len(dataset) * self.train_test_split)]
        else:
            self.data = dataset[int(len(dataset) * self.train_test_split):]

    def LSTMSequenceGen(self, max_seq_length, embedding_dim, word2vecModel, train=True):
        '''
        From the original dataset, runs every word from a document through Word2Vec and
        stacks the sequence of numpy arrays to fit the max sequence length, padding with
        zeros if neccessary. Can only be called after Word2Vec model has been trained.

        :params  dataset: list of samples ['Text', bool]
                 max_seq_length: largest sequence length in dataset
                 num_samples: number of samples in dataset
                 embedding_dim: size of embedding length for Word2Vec
                 word2vecModel: trained model for Word2Vec
                 train: bool, whether passed dataset is for training or not
        :return  sequence: numpy array (sequence_length x 5)
        '''

        num_samples = self.__len__()
        dir_name = os.path.dirname(os.path.abspath(inspect.getfile
                                                   (inspect.currentframe())))
        if train:
            file_name = '../log/dataset_train.npz'
        else:
            file_name = '../log/dataset_test.npz'

        if not os.path.exists(os.path.join(dir_name, file_name)):
            self.result = np.zeros((num_samples, max_seq_length, embedding_dim))

            data = [sample[0].split(' ') for sample in self.data]
            vecs = [word2vecModel.vectorize_sentence(words) for words in data]
            stack = [np.stack(vecs[i], axis=0) for i in range(len(data))]

            for i in range(len(stack)):
                self.result[i, :stack[i].shape[0], :stack[i].shape[1]] = stack[i]

            targets = [sample[1] for sample in self.data]
            self.targets = np.asarray(targets)

            np.savez(os.path.join(dir_name, file_name), data=self.result,
                     targets=self.targets)
        else:
            loader = np.load(os.path.join(dir_name, file_name))
            self.result = loader['data.npy']
            self.targets = loader['targets.npy']

    def __getitem__(self, index):
        # return self.data[index][0], self.data[index][1]
        return self.result[index], self.targets[index]

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    dataset = Dataset()
    print(dataset.all_samples[1])
