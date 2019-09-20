import json
import pandas
import seaborn as sns
import matplotlib.pyplot as plt

from inDexDa.utils.visualization import viz
from inDexDa.lib.normalize_text import Normalize
# from scipy.sparse import coo_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


def get_top_n_words(corpus, n=None):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:n]


def get_top_n2_words(corpus, n=None):
    vec1 = CountVectorizer(ngram_range=(2, 2),
                           max_features=2000).fit(corpus)
    bag_of_words = vec1.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in
                  vec1.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:n]


def get_top_n3_words(corpus, n=None):
    vec1 = CountVectorizer(ngram_range=(3, 3), max_features=2000).fit(corpus)
    bag_of_words = vec1.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in
                  vec1.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:n]


def plot_n_gram_frequency(corpus):
    # Convert most freq words to dataframe for plotting bar plot
    top_words = get_top_n_words(corpus, n=20)
    top_df = pandas.DataFrame(top_words)
    top_df.columns = ["Word", "Freq"]

    # Convert most freq bi-grams to dataframe for plotting bar plot
    top2_words = get_top_n2_words(corpus, n=20)
    top2_df = pandas.DataFrame(top2_words)
    top2_df.columns = ["Bi-gram", "Freq"]

    # Convert most freq tri-grams to dataframe for plotting bar plot
    top3_words = get_top_n3_words(corpus, n=20)
    top3_df = pandas.DataFrame(top3_words)
    top3_df.columns = ["Tri-gram", "Freq"]

    plt.figure('Mono-Gram Frequency')
    g = sns.barplot(x="Word", y="Freq", data=top_df)
    g.set_xticklabels(g.get_xticklabels(), rotation=30)

    plt.figure('Bi-Gram Frequency')
    h = sns.barplot(x="Bi-gram", y="Freq", data=top2_df)
    h.set_xticklabels(h.get_xticklabels(), rotation=45)

    plt.figure('Tri-Gram Frequency')
    j = sns.barplot(x="Tri-gram", y="Freq", data=top3_df)
    j.set_xticklabels(j.get_xticklabels(), rotation=45)

    plt.show()


def sort_coo(comatrix):
    tuples = zip(comatrix.col, comatrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)


def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""

    # use only topn items from vector
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []

    # word index and corresponding tf-idf score
    for idx, score in sorted_items:

        # keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    # create a tuples of feature,score
    # results = zip(feature_vals,score_vals)
    results = {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]] = score_vals[idx]

    return results


'''
#########################################################################################
############################### FOR FIRST 50 ARXIV ENTRIES###############################
#########################################################################################
'''
# with open('PaperScraper/data/arxiv/papers.json', 'r') as f:
#     file = json.load(f)

# abstracts = []
# for i in range(0, 1000):
#     abstract = file[i]['Abstract']
#     abstracts.append(abstract)

'''
#########################################################################################
############################### FOR TRUE POSITIVE DATASET ###############################
#########################################################################################
'''
with open('data/dataset.json', 'r') as f:
    file = json.load(f)

abstracts = []
for entry in file:
    abstract = entry['Abstract']
    abstracts.append(abstract)

corpus = []
for abstract in abstracts:
    normalize = Normalize(abstract)
    text = normalize.normalized_text
    corpus.append(' '.join(text))

plot_n_gram_frequency(corpus)

corpus = ' '.join(corpus)
max_words = len(corpus.split())
viz(corpus, max_words)


cv = CountVectorizer(max_df=0.8, min_df=0.001, max_features=10000, ngram_range=(1, 3))
X = cv.fit_transform(corpus)

tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
tfidf_transformer.fit(X)

# get feature names
feature_names = cv.get_feature_names()

for i in range(0, 20):
    # fetch document for which keywords needs to be extracted
    doc = corpus[i]

    # generate tf-idf for the given document
    tf_idf_vector = tfidf_transformer.transform(cv.transform([doc]))

    sorted_items = sort_coo(tf_idf_vector.tocoo())
    # extract only the top n; n here is 10
    keywords = extract_topn_from_vector(feature_names, sorted_items, 5)

    with open('inDexDa/log/idx_test.txt', 'a+') as f:
        f.write("Abstract:")
        f.write(doc + '\n')
        f.write("\nKeywords:\n")
        for k in keywords:
            f.write('{}: {}\n'.format(k, keywords[k]))
