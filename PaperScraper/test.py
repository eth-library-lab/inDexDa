import re
import nltk
import inflect
import contractions
import re, string, unicodedata

from bs4 import BeautifulSoup
from unidecode import unidecode
from nltk.corpus import stopwords
from nltk import word_tokenize, sent_tokenize
from nltk.stem import LancasterStemmer, WordNetLemmatizer


def replace_contractions(text):
    """Replace contractions in string of text"""
    return contractions.fix(text)

def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    if isinstance(words, list):
        new_words = []
        for word in words:
            # new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
            # new_words.append(new_word)
            new_words.append(unidecode(word))
        return new_words
    elif isinstance(words, str):
        return unidecode(words)

def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

def replace_numbers(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words

def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words

def stem_words(words):
    """Stem words in list of tokenized words"""
    stemmer = LancasterStemmer()
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems

def lemmatize_verbs(words):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas

def normalize(words):
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = replace_numbers(words)
    words = remove_stopwords(words)
    return words

def stem_and_lemmatize(words):
    stems = stem_words(words)
    lemmas = lemmatize_verbs(words)
    return stems, lemmas



abstract1 = "We describe our two new datasets with images described by humans. Both the datasets were collected using Amazon Mechanical Turk, a crowdsourcing platform. The two datasets contain significantly more descriptions per image than other existing datasets. One is based on a popular image description dataset called the UIUC Pascal Sentence Dataset, whereas the other is based on the Abstract Scenes dataset con- taining images made from clipart objects. In this paper we describe our interfaces, analyze some properties of and show example descriptions from our two datasets."
abstract2 = "A dataset is crucial for model learning and evaluation. Choosing the right dataset to use or making a new dataset requires the knowledge of those that are available. In this work, we provide that knowledge, by reviewing twenty datasets that were published in the recent six years and that are directly related to object manipulation. We report on modalities, activities, and annotations for each individual dataset and give our view on its use for object manipulation. We also compare the datasets and summarize them. We conclude with our suggestion on future datasets."
abstract3 = "Existing datasets for natural language inference (NLI) have propelled research on language understanding. We propose a new method for automatically deriving NLI datasets from the growing abundance of large-scale question answering datasets. Our approach hinges on learning a sentence transformation model which converts question-answer pairs into their declarative forms. Despite being primarily trained on a single QA dataset, we show that it can be successfully applied to a variety of other QA resources. Using this system, we automatically derive a new freely available dataset of over 500k NLI examples (QA-NLI), and show that it exhibits a wide range of inference phenomena rarely seen in previous NLI datasets."
abstract4 = "In Kimmo K\u00e4rkk\u00e4inen this paper, we disseminate a new handwritten digits-dataset, termed Kannada-MNIST, for the Kannada script, that can potentially serve as a direct drop-in replacement for the original MNIST dataset. In addition to this dataset, we disseminate an additional real world handwritten dataset (with $10k$ images), which we term as the Dig-MNIST dataset that can serve as an out-of-domain test dataset. We also duly open source all the code as well as the raw scanned images along with the scanner settings so that researchers who want to try out different signal processing pipelines can perform end-to-end comparisons. We provide high level morphological comparisons with the MNIST dataset and provide baselines accuracies for the dataset disseminated. The initial baselines obtained using an oft-used CNN architecture ($96.8\\%$ for the main test-set and $76.1\\%$ for the Dig-MNIST test-set) indicate that these datasets do provide a sterner challenge with regards to generalizability than MNIST or the KMNIST datasets. We also hope this dissemination will spur the creation of similar datasets for all the languages that use different symbols for the numeral digits."
abstract5 = "Machine learning tasks such as optimizing the hyper-parameters of a model for a new dataset or few-shot learning can be vastly accelerated if they are not done from scratch for every new dataset, but carry over findings from previous runs. Meta-learning makes use of features of a whole dataset such as its number of instances, its number of predictors, the means of the predictors etc., so called meta-features, dataset summary statistics or simply dataset characteristics, which so far have been hand-crafted, often specifically for the task at hand. More recently, unsupervised dataset encoding models based on variational auto-encoders have been successful in learning such characteristics for the special case when all datasets follow the same schema, but not beyond. In this paper we design a novel model, Dataset2Vec, that is able to characterize datasets with a latent feature vector based on batches and thus is able to generalize beyond datasets having the same schema to arbitrary (tabular) datasets. To do so, we employ auxiliary learning tasks on batches of datasets, esp. to distinguish batches from different datasets. We show empirically that the meta-features collected from batches of similar datasets are concentrated within a small area in the latent space, hence preserving similarity. We also show that using the dataset characteristics learned by Dataset2Vec in a state-of-the-art hyper-parameter optimization model outperforms the hand-crafted meta-features that have been used in the hyper-parameter optimization literature so far. As a result, we advance the current state-of-the-art results for hyper-parameter optimization."
abstracts = [abstract1, abstract2, abstract3, abstract4, abstract5]


for abstract in abstracts:
    sample = remove_non_ascii(abstract)
    sample = re.sub('[^0-9a-zA-Z]+', ' ', sample)
    sample = replace_contractions(sample)
    words = nltk.word_tokenize(sample)
    words = normalize(words)
    _, lemmas = stem_and_lemmatize(words)

    with open('log/test.txt', 'a+') as f:
        f.write("NEW ABSTRACT\n")
        f.write(' '.join(lemmas))
        f.write('\n')
        f.write(abstract)
        f.write('\n')
        f.write('\n')


