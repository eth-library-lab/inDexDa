import os
import nltk
import json

from NLP.utils.normalize_text import Normalize


class PreprocessForBert():
    def __init__(self, min_thresh=10, max_thresh=None, train_test_split=0.8):
        '''
        Reads dataset from a json file. Normalizes and tokenizes the dataset as well as
         removes uncommon words (to better handle them in machine learning applications).

        :params  min_thresh(int)                  : minimum word count threshold
                 max_thresh(int)                  : maximum word count threshold
                 train_test_split(float)          : train test split
        '''
        self.min_thresh = min_thresh
        self.max_thresh = max_thresh
        self.train_test_split = train_test_split
        self.current_dir = os.path.dirname(os.path.abspath(__file__))

    def readCorpus(self, fpos, fneg):
        '''
        Reads the files containing both the positive and negative training examples and
        combines them into a format used for training the network.

        :params  fpos: file string for positive dataset
                 fneg: file string for negative dataset

        :return  positives: positive training examples
                 negatives: negative training examples
        '''

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

        # Read negative sample dataset from a json file
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

        return positives, negatives

    def normalizeCorpus(self, corpus):
        '''
        Takes the given corpus and processes it to remove certain words and non-ASCII
        characters. Also tokenizes the words.

        :params  corpus: list of strings, each string a new abstract from a paper
        :return  norm_corpus: normalized corpus in same format
        '''

        # Normalize and tokenize corpus using nltk and other NLP packages
        norm_corpus = []
        for doc in corpus:
            normalize = Normalize(doc[0], removeStopWords=False, tokenize=False,
                                  toLower=False, removePunctuation=False,
                                  removeNonASCII=True, removeContradictions=False,
                                  denoise=False, stem=False, lemmatize=False)
            norm_corpus .append([normalize.normalized_text, doc[1]])
        return norm_corpus

    def removeSpecializedWords(self, corpus):
        '''
        Removes specialized words from a corpus that only appear less than the minimum
        threshold (specified in class variable).

        :params  corpus: list of strings containing the abstracts of papers
        :return  corpus: corpus with speciailzed words replaces with 'unknown'
        '''

        # Make a word count and replace uncommon words with the keyword unknown
        vocab = []
        for doc in corpus:
            vocab.extend(doc[0])
        freq = nltk.probability.FreqDist(vocab)

        # Dict of words which appear less than x times in corpus. Replace all instances
        #  of these words within the corpus with the keyword unknown.
        uncommon = dict(filter(lambda x: x[1] <= self.min_thresh, freq.items()))
        for k, v in uncommon.items():
            uncommon[k] = 'unknown'

        corpus[:] = [[' '.join([uncommon.get(word, word) for word in doc[0]]), doc[1]] for doc in corpus]
        return corpus

    def makeDirs(self):
        '''
        Creates the required training and testing data directories in order for the ktrain
        BERT network to be run. For more information about why this is necessary, please
        read the inDexDa manual.

        :params  N/A
        '''

        if not os.path.exists(self.current_dir, '../data/bert_data'):
            os.mkdir(os.path.join(self.current_dir, '../data/bert_data'))
        if not os.path.exists(self.current_dir, '../data/bert_data/train'):
            os.mkdir(os.path.join(self.current_dir, '../data/bert_data/train'))
        if not os.path.exists(self.current_dir, '../data/bert_data/test'):
            os.mkdir(os.path.join(self.current_dir, '../data/bert_data/test'))
        if not os.path.exists(self.current_dir, '../data/bert_data/train/0'):
            os.mkdir(os.path.join(self.current_dir, '../data/bert_datatrain/0'))
        if not os.path.exists(self.current_dir, '../data/bert_data/train/1'):
            os.mkdir(os.path.join(self.current_dir, '../data/bert_data/train/1'))
        if not os.path.exists(self.current_dir, '../data/bert_data/test/0'):
            os.mkdir(os.path.join(self.current_dir, '../data/bert_data/test/0'))
        if not os.path.exists(self.current_dir, '../data/bert_data/test/1'):
            os.mkdir(os.path.join(self.current_dir, '../data/bert_data/test/1'))

    def saveBertSamples(self, train_pos, test_pos, train_neg, test_neg):
        '''
        Segments the positive and negative training examples into training and testing
        categories based on train/test split. Saves each example as a txt file in the
        specific directory.

        :params  train_pos: positive examples used for training
                 test_pos: positive examples used for testing
                 train_neg: negaitve examples used for training
                 test_neg: negative examples used for testing
        '''
        for idx, sample in enumerate(train_pos):
            ix = idx + 1
            file_name = os.path.join(self.current_dir,
                                     "../data/bert_data/train/1/{:04d}.txt".format(ix))
            with open(file_name, 'w') as f:
                f.write(sample['Text'])

        for idx, sample in enumerate(test_pos):
            file_name = os.path.join(self.current_dir,
                                     "../data/bert_data/test/1/{:04d}.txt".format(ix))
            with open(file_name, 'w') as f:
                f.write(sample['Text'])

        for idx, sample in enumerate(train_neg):
            file_name = os.path.join(self.current_dir,
                                     "../data/bert_data/train/0/{:04d}.txt".format(ix))
            with open(file_name, 'w') as f:
                f.write(sample['Text'])

        for idx, sample in enumerate(test_neg):
            file_name = os.path.join(self.current_dir,
                                     "../data/bert_data/test/0/{:04d}.txt".format(ix))
            with open(file_name, 'w') as f:
                f.write(sample['Text'])

    def processForTrainingBert(self):
        '''
        Preprocesses all the required data to run the BERT network. To load the network
        it needs to be fed a training and testing set, so we use the one provided with
        inDexDa. Another dataset can be used as well, but the data must be segmented
        into two json files: one for positive classification examples, and one with
        negative classification examples. See original files for formatting details.

        Preprocessing takes these files and creates the necessary data segmentation
        structure.

        :params  N/A
        '''
        print('Processing data to train BERT network...')

        # Sets up the positive and negative corpus' and removes non-ASCII characters
        fpos = os.path.join(self.current_dir, '../data/positive_samples.json')
        fneg = os.path.join(self.current_dir, '../data/negative_samples.json')
        positives, negatives = self.readCorpus(fpos, fneg)

        # positives = self.removeSpecializedWords(self.normalizeCorpus(positives))
        # negatives = self.removeSpecializedWords(self.normalizeCorpus(negatives))
        positives = self.normalizeCorpus(positives)
        negatives = self.normalizeCorpus(negatives)

        # Sets up the testing and training sets
        train_pos = positives[:len(positives) * self.train_test_split]
        test_pos = positives[len(positives) * self.train_test_split:]

        train_neg = negatives[:len(negatives) * self.train_test_split]
        test_neg = negatives[len(negatives) * self.train_test_split:]

        self.makeDirs()
        self.saveBertSamples(train_pos, test_pos, train_neg, test_neg)


class PreprocessScrapedData():
    '''
    Gathers the scraped papers from the archives used, combines them into one file, and
        deletes the original files to save memory.

    :params     archivesUsed(list)      : list of archives used in scraping
                current_dir(int)        : current directory
                datadirs(list)          : list of archive directories for scraped papers
    '''
    def __init__(self, archivesUsed):
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.datadirs = self.getDataDirs(archivesUsed)

    def getDataDirs(self, archivesUsed):
        # Gets directories for scraped papers fro mthe used archives

        datadirs = []
        for archive in archivesUsed:
            archive_dir = os.path.join('../PaperScraper/data', archive, 'papers.json')
            datadirs.append(os.path.join(self.current_dir, archive_dir))

        return datadirs

    def transferData(self):
        # Moves all scraped papers to one file and deletes the originals.

        output_file = os.path.join(self.current_dir, '../data/results.json')

        data = []
        for datadir in self.datadirs:
            with open(datadir, 'r') as f:
                contents = f.read()
                papers = json.loads(contents)

                data.extend(papers)

        with open(output_file, 'w') as f:
            json.dump(data, f, indent=4)

        for datadir in self.datadirs:
            os.remove(datadir)


# if __name__ == '__main__':
    # print('Processing the documents. This may take a couple minutes ...')
    # archivesUsed = ['arxiv', 'sciencedirect']
    # preprocess = PreprocessScrapedData(archivesUsed)
    # preprocess.transferData()

    # preprocess = PreprocessForBert()
    # preprocess.processForBert()
