import os
import torch.nn as nn
import torch.nn.functional as F

from gensim.models.doc2vec import Doc2Vec, TaggedDocument


class NLPModel():
    def __init__(self, epochs=100, method='PVDM', vec_size=20):
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.epochs = epochs
        self.vec_size = vec_size
        self.alpha = 0.025
        if method == 'PVDM':
            self.dm = 1
        else:
            self.dm = 0

    def train(self, dataset):
        '''
        Trains Dic2Vec model using both positive and negative examples

        :params  dataset: list of strings, each string a document
        '''
        documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(dataset)]

        if not os.path.exists(os.path.join(self.current_dir, '../log/d2v.model')):
            self.model = Doc2Vec(alpha=self.alpha,
                                 min_count=1,
                                 dm=self.dm,
                                 workers=4,
                                 window=4,
                                 vector_size=self.vec_size)
            self.model.build_vocab(documents)
            self.model.train(documents,
                             total_examples=self.model.corpus_count,
                             epochs=self.epochs)

            self.model.save(os.path.join(self.current_dir, "../log/d2v.model"))
            print("Doc2Vec Model Saved ...")

    def doc_vector(self, doc):
        '''
        Given a document, returns the encoded vector

        :params  doc: list of strings
        :return  numpy array of size (1, 50)
        '''
        try:
            path = os.path.join(self.current_dir, "../log/d2v.model")
            self.model = Doc2Vec.load(path)
            return self.model.infer_vector(doc)
        except AttributeError:
            print('Error: model could not be loaded')


class DocClassifier(nn.Module):
    def __init__(self):
        super(DocClassifier, self).__init__()
        self.fc1 = nn.Linear(50, 25)
        self.fc2 = nn.Linear(25, 10)
        self.fc3 = nn.Linear(10, 2)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
