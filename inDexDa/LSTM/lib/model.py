import torch
import torch.nn as nn
import torch.nn.functional as F

from gensim.models import Word2Vec


class Word2VecModel():
    def __init__(self, epochs=100, window=5, min_count=1, size=5):
        self.epochs = epochs
        self.window = window
        self.min_count = min_count
        self.size = size

    def train(self, dataset):
        '''
        Trains Word2Vec model for word embeddings which are used in the LSTM network

        :params  dataset: list of strings, each string a document containing many words
        '''
        self.model = Word2Vec(size=self.size,
                              window=self.window,
                              min_count=self.min_count,
                              workers=4)
        self.model.build_vocab(dataset)
        self.model.train(dataset,
                         total_examples=self.model.corpus_count,
                         epochs=self.epochs)
        self.model.save("/home/parker/code/datadex/inDexDa/LSTM/log/w2v.model")
        print("Word2Vec Model Saved ...")

    def vectorize_sentence(self, sentence):
        '''
        Given a word, returns a vectorized encoding

        :params  sentence: string or list of strings
        :return  numpy array of size (1, 5)
        '''
        if isinstance(sentence, list):
            try:
                path = "/home/parker/code/datadex/inDexDa/LSTM/log/w2v.model"
                self.model = Word2Vec.load(path)
                return [self.model.wv[word] for word in sentence]
            except AttributeError:
                print('Error: model could not be loaded')
            except FileNotFoundError:
                print('Error: saved model filepath does not exist')
        else:
            try:
                path = "/home/parker/code/datadex/inDexDa/LSTM/log/w2v.model"
                self.model = Word2Vec.load(path)
                return self.model.wv[sentence]
            except AttributeError:
                print('Error: model could not be loaded')
            except FileNotFoundError:
                print('Error: saved model filepath does not exist')


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, batch_size, output_dim, seq_len):
        super(LSTMClassifier, self).__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.layers = 1
        self.batch_size = batch_size

        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim)
        self.f1 = nn.Linear(self.hidden_dim, 2)

    def forward(self, x):
        x = torch.transpose(x, 0, 1)
        x, _ = self.lstm(x)
        x = self.f1(x)

        return x[-1]
