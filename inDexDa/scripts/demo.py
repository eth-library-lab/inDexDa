import os
import argparse
from gensim.models import Doc2Vec
from lib.labeled_doc import LabeledLineSentence
from utils.train_test_setup import train_test_sets
from sklearn.linear_model import LogisticRegression


parser = argparse.ArgumentParser()
parser.add_argument('--num_positives', type=int, default=600, help='database used')
parser.add_argument('--num_negatives', type=int, default=5000, help='field of research')
parser.add_argument('--vocab_size', type=int, default=250, help='field of research')
parser.add_argument('--epochs', type=int, default=20, help='field of research')
options = parser.parse_args()
options.samples = options.num_negatives + options.num_positive

'''
Set up .txt sources for abstracts and their associated labels
'''
################# TO DO ##################
sources = {'test-neg.txt': 'TEST_NEG', 'test-pos.txt': 'TEST_POS',
           'train-neg.txt': 'TRAIN_NEG', 'train-pos.txt': 'TRAIN_POS',
           'train-unsup.txt': 'TRAIN_UNS'}
sentences = LabeledLineSentence(sources)
##########################################


'''
Build Doc2Vec model using gensim for word embeddings
'''
model = Doc2Vec(min_count=1, window=10, size=options.vocab_size, sample=1e-4, negative=5,
                workers=2)
model.build_vocab(sentences.to_array())

if not os.path.exists('../data/inDexDa.d2v'):
    for epoch in range(options.epochs):
        model.train(sentences.sentences_perm())

    model.save('../data/inDexDa.d2v')
else:
    model = Doc2Vec.load('../data/inDexDa.d2v')


'''
Make classifer to determine whether a dataset was created
'''
train, test = train_test_sets(model, options)

classifier = LogisticRegression()
classifier.fit(train.arrays, train.labels)
print(classifier.score(test.arrays, test.labels))
