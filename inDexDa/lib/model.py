import os
import sys
import nltk
import math
import json
import random
import gensim
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F

from normalize_text import Normalize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument



class Doc2Vec_Dataset():
    def __init__(self, min_thresh=5, max_thresh=None, train_test_split=0.8):
        '''
        Reads dataset from a json file. Normalizes and tokenizes the dataset as well as
         removes uncommon words (to better handle them in machine learning applications).

        :params  train(list of lists)     : samples for training
                 test(list of lists)      : samples for testing
                 min_thresh(int)          : minimum word count threshold
                 max_thresh(int)          : maximum word count threshold
                 train_test_split(float)  : train test split
        '''
        self.train = []
        self.test = []
        self.min_thresh = min_thresh
        self.max_thresh = max_thresh
        self.train_test_split = train_test_split

        self.process()

    def process(self):
        # Read, normalize/tokenize, and trim dataset. Partition dataset for training
        dataset = self.normalize_corpus(self.read_corpus())
        dataset = self.remove_specialized_words(dataset)
        self.partition_dataset(dataset)

    def read_corpus(self, fname='/home/parker/code/datadex/data/dataset.json'):
        # Read dataset from a json file
        dataset = []
        with open(fname, 'r') as f:
            data = f.read()
            try:
                raw = json.loads(data)
                for paper in raw:
                    dataset.append(paper['Abstract'])
                return dataset
            except ValueError:
                print('Not able to parse json file to dictionary.\n')

    def normalize_corpus(self, corpus):
        # Normalize and tokenize dataset using nltk and other NLP packages
        norm_corpus = []
        for doc in corpus:
            normalize = Normalize(doc, removeStopWords=True, tokenize=True)
            norm_corpus .append(normalize.normalized_text)
        return norm_corpus

    def remove_specialized_words(self, dataset):
        # Make a word count and replace uncommon words with the keyword unknown
        vocab = []
        for doc in dataset:
            vocab.extend(doc)
        freq = nltk.probability.FreqDist(vocab)

        # Dict of words which appear less than 5 times in dataset. Replace all instances
        #  of these words within the corpus with the keyword unknown.
        uncommon = dict(filter(lambda x: x[1] <= self.min_thresh, freq.items()))
        for k, v in uncommon.items():
            uncommon[k] = 'unknown'

        dataset[:] = [[uncommon.get(word, word) for word in doc] for doc in dataset]
        return dataset

    def partition_dataset(self, dataset):
        # Split dataset into testing and training samples
        train_samples = math.ceil(len(dataset) * self.train_test_split)

        self.train = [dataset[i] for i in range(train_samples)]
        self.test = [dataset[i] for i in range(train_samples, len(dataset))]


class Classifier_Dataset(data.Dataset):
    def __init__(self, positive_examples, negative_examples):
        plength = len(positive_examples)
        nlength = len(negative_examples)

        self.target[: plength] = 1
        self.target[plength: plength + nlength] = 0
        self.n_samples = self.data.shape[0]
        self.data = self.process()

    def __len__(self):   # Length of the dataset.
        return self.n_samples

    def __getitem__(self, index):   # Function that returns one point and one label.
        return torch.Tensor(self.data[index]), torch.Tensor(self.target[index])

    def process(self):
        # Read, normalize/tokenize, and trim dataset. Partition dataset for training
        dataset = []
        dataset.extend(positive_examples)
        dataset.extend(negative_examples)

        dataset = self.normalize_corpus(dataset)
        dataset = self.remove_specialized_words(dataset)

        return dataset

    def normalize_corpus(self, corpus):
        # Normalize and tokenize dataset using nltk and other NLP packages
        norm_corpus = []
        for doc in corpus:
            normalize = Normalize(doc, removeStopWords=True, tokenize=True)
            norm_corpus .append(normalize.normalized_text)
        return norm_corpus

    def remove_specialized_words(self, dataset):
        # Make a word count and replace uncommon words with the keyword unknown
        vocab = []
        for doc in dataset:
            vocab.extend(doc)
        freq = nltk.probability.FreqDist(vocab)

        # Dict of words which appear less than 5 times in dataset. Replace all instances
        #  of these words within the corpus with the keyword unknown.
        uncommon = dict(filter(lambda x: x[1] <= self.min_thresh, freq.items()))
        for k, v in uncommon.items():
            uncommon[k] = 'unknown'

        dataset[:] = [[uncommon.get(word, word) for word in doc] for doc in dataset]
        return dataset


class NLPModel():
    def __init__(self, epochs=100, method='PVDM'):
        self.epochs = epochs
        self.vec_size = 20
        self.alpha = 0.025
        if method=='PVDM':
            self.dm = 1
        else:
            self.dm = 0

    def train(self, dataset):
        documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(dataset.train)]

        if not os.path.exists('.log/d2v.model'):
            self.model = Doc2Vec(alpha=self.alpha,
                                 min_count=1,
                                 dm=self.dm,
                                 workers=4,
                                 window=5,
                                 vector_size=50)
            self.model.build_vocab(documents)
            self.model.train(documents,
                             total_examples=self.model.corpus_count,
                             epochs=self.epochs)

            self.model.save("log/d2v.model")
            print("Model Saved")
        else:
            self.model= Doc2Vec.load("d2v.model")

        test = self.model.infer_vector(dataset.test[0])
        print("Infered Test Vector: {}".format(test))

    def doc_vectors(self):
        print()


class DocClassifier(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(50, 25)
        self.fc2 = nn.Linear(25, 10)
        self.fc3 = nn.Linear(10, 3)

    def forward(self, x):
           x = F.relu(self.fc1(x))
           x = F.relu(self.fc2(x))
           x = F.relu(self.fc3(x))
           return F.log_softmax(x)


dataset = Doc2Vec_Dataset()
model = NLPModel(epochs=100, method='PVDM')
model.train(dataset)

data = Classifier_Dataset(p_dataset, n_dataset)
dataloader = data.DataLoader(data, batch_size=1, num_workers=8)

for epoch in range(20):
    for k, (data, target) in enumerate(dataloader):
        # Definition of inputs as variables for the net.
        # requires_grad is set False because we do not need to compute the
        # derivative of the inputs.
        # data   = Variable(data,requires_grad=False)
        # target = Variable(target.long(),requires_grad=False)

        # Set gradient to 0.
        optimizer.zero_grad()
        # Feed forward.
        pred = model(data)
        # Loss calculation.
        loss = criterium(pred, target)
        # Gradient calculation.
        loss.backward()
        # Model weight modification based on the optimizer.
        optimizer.step()

    print('Epoch : {} of {}'.format(epoch, 20))

