'''
Text normalization for Doc2Vec vocabulary
'''
 wnl = nltk.WordNetLemmatizer()
>>> [wnl.lemmatize(t) for t in tokens]
