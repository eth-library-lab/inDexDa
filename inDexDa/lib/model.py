import os
import sys
import nltk
import math
import json
import random
import gensim
import collections

from normalize_text import Normalize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument



class Dataset():
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


dataset = Dataset()
model = NLPModel(epochs=100, method='PVDM')
model.train(dataset)
