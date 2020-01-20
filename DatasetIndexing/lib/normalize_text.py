import re
import nltk
import contractions

from unidecode import unidecode
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer


class Normalize():
    def __init__(self, text, toLower=True, removePunctuation=True, removeNonASCII=True,
                 removeContradictions=True, denoise=True, removeStopWords=True,
                 stem=False, lemmatize=True, tokenize=True):
        '''
        Takes string of words to normalize, tokenize, and/or stemmatize

        :params  text (str): string of words
                 normalized_text (list): list of normalized words from text
        '''
        if not isinstance(text, str):
            raise TypeError('must be str, not list')

        self.stem = stem
        self.denoise = denoise
        self.toLower = toLower
        self.tokenize = tokenize
        self.lemmatize = lemmatize
        self.removeNonASCII = removeNonASCII
        self.removeStopWords = removeStopWords
        self.removePunctuation = removePunctuation
        self.removeContradictions = removeContradictions

        self.text = text
        self.normalized_text = self.__process()

    def __remove_non_ascii(self, words):
        """Remove non-ASCII characters from list of tokenized words"""
        if isinstance(words, list):
            new_words = []
            for word in words:
                new_words.append(unidecode(word))
            return new_words
        elif isinstance(words, str):
            return unidecode(words)

    def __replace_contractions(self, text):
        """Replace contractions in string of text"""
        return contractions.fix(text)

    def __denoise(self, words):
        """Remove numbers, urls, and symbols from string"""
        words = re.sub(r'http\S+', '', words)         # Remove urls
        words = re.sub(r'\d+', '', words)             # Remove numbers
        words = re.sub('[^0-9a-zA-Z]+', ' ', words)   # Remove symbols
        return words

    def __to_lowercase(self, words):
        """Convert all characters to lowercase from list of tokenized words"""
        if isinstance(words, list):
            new_words = []
            for word in words:
                new_word = word.lower()
                new_words.append(new_word)
            return new_words
        elif isinstance(words, str):
            return words.lower().split(' ')

    def __remove_punctuation(self, words):
        """Remove punctuation from list of tokenized words"""
        if isinstance(words, list):
            new_words = []
            for word in words:
                new_word = re.sub(r'[^\w\s]', '', word)
                if new_word != '':
                    new_words.append(new_word)
            return new_words
        elif isinstance(words, str):
            return re.sub(r'[^\w\s]', '', words).split(' ')

    def __remove_stopwords(self, words):
        """Remove stop words from list of tokenized words"""
        if isinstance(words, list):
            new_words = []
            for word in words:
                if word not in stopwords.words('english'):
                    new_words.append(word)
            return new_words
        elif isinstance(words, str):
            words = words.split(' ')
            new_words = []
            for word in words:
                if word not in stopwords.words('english'):
                    new_words.append(word)
            return new_words

    def __lemmatize_verbs(self, words):
        """Lemmatize verbs in list of tokenized words"""
        if isinstance(words, str):
            words = words.split(' ')

        lemmatizer = WordNetLemmatizer()
        lemmas = []
        for word in words:
            lemma = lemmatizer.lemmatize(word, pos='v')
            lemmas.append(lemma)
        return lemmas

    def __stem_words(self, words):
        """Stem words in list of tokenized words"""
        if isinstance(words, str):
            words = words.split(' ')

        stemmer = LancasterStemmer()
        stems = []
        for word in words:
            stem = stemmer.stem(word)
            stems.append(stem)
        return stems

    def __process(self):
        sample = self.text

        if self.removeNonASCII:
            sample = self.__remove_non_ascii(sample)

        if self.removeContradictions:
            sample = self.__replace_contractions(sample)

        if self.denoise:
            sample = self.__denoise(sample)

        if self.tokenize:
            sample = nltk.word_tokenize(sample)

        if self.toLower:
            sample = self.__to_lowercase(sample)

        if self.removePunctuation:
            sample = self.__remove_punctuation(sample)

        if self.removeStopWords:
            sample = self.__remove_stopwords(sample)

        if self.lemmatize:
            sample = self.__lemmatize_verbs(sample)

        if self.stem:
            sample = self.__stem_words(sample)

        return sample

if __name__ == '__main__':
    link = "http://kitchen.cs.cmu.edu/http://ai.stanford.edu/\u02dcalireza"
    linknew = Normalize(link, toLower=False, removePunctuation=False, removeNonASCII=True,
                        removeContradictions=False, denoise=False, removeStopWords=False,
                        stem=False, lemmatize=False, tokenize=False)
    print(linknew.normalized_text)
