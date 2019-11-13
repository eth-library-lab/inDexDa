import nltk
import json
from normalize_text import Normalize


class Preprocess():
    def __init__(self, train=True, min_thresh=15, max_thresh=None,
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
        self.data = []
        self.alldata = []
        self.train = train
        self.min_thresh = min_thresh
        self.max_thresh = max_thresh
        self.train_test_split = train_test_split

    def process(self):
        # Read positive and negative samples from their files into the same set
        fpos = '/home/parker/code/datadex/data/positive_samples.json'
        fneg = '/home/parker/code/datadex/data/negative_samples.json'
        dataset = self.read_corpus(fpos, fneg)

        # Normalize/Tokenize/Remove specialized words from the set
        dataset = self.remove_specialized_words(self.normalize_corpus(dataset))
        self.save(dataset)

    def read_corpus(self, fpos, fneg):
        # Read positive sample dataset from a json file
        dataset = []
        with open(fpos, 'r') as f:
            data = f.read()
            try:
                raw = json.loads(data)
                for paper in raw:
                    dataset.append([paper['Abstract'], 1])
                positives = dataset
            except ValueError:
                print('Not able to parse json file to dictionary.\n')

        dataset = []
        with open(fneg, 'r') as f:
            data = f.read()
            try:
                raw = json.loads(data)
                for paper in raw:
                    dataset.append([paper['Abstract'], 0])
                negatives = dataset
            except ValueError:
                print('Not able to parse json file to dictionary.\n')

        return positives + negatives

    def normalize_corpus(self, corpus):
        # Normalize and tokenize corpus using nltk and other NLP packages
        norm_corpus = []
        for doc in corpus:
            normalize = Normalize(doc[0], removeStopWords=True, tokenize=True)
            norm_corpus .append([normalize.normalized_text, doc[1]])
        return norm_corpus

    def remove_specialized_words(self, corpus):
        # Make a word count and replace uncommon words with the keyword unknown
        vocab = []
        for doc in corpus:
            vocab.extend(doc[0])
        freq = nltk.probability.FreqDist(vocab)

        # Dict of words which appear less than 5 times in corpus. Replace all instances
        #  of these words within the corpus with the keyword unknown.
        uncommon = dict(filter(lambda x: x[1] <= self.min_thresh, freq.items()))
        for k, v in uncommon.items():
            uncommon[k] = 'unknown'

        corpus[:] = [[' '.join([uncommon.get(word, word) for word in doc[0]]), doc[1]] for doc in corpus]
        return corpus

    def save(self, dataset):
        dataset_dict = []
        for item in dataset:
            dataset_dict.append({'Text': item[0], 'Target': item[1]})

        with open('/home/parker/code/datadex/NaturalLanguageProcessing/LSTM/log/dataset.json', 'w') as f:
            json.dump(dataset_dict, f, indent=4)


if __name__ == '__main__':
    print('Processing the documents. This may take a couple minutes ...')
    preprocess = Preprocess()
    preprocess.process()
