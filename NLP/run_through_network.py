import os
import json

# from NLP.preprocess import NormalizePapers
from NLP.BERT.lib.test import test as BertTest


def runThroughNetwork():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    datadir = os.path.join(current_dir, '../data/eval.json')

    try:
        BertTest(datadir)
    except Exception as bert_error:
        raise Exception(bert_error)

