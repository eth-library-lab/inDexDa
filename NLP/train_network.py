import os
from NLP.BERT.lib.train import train as BertTrain


def trainNetwork():
    '''
    Trains the BERT network using the training data in inDexDa/data/bert_data.
    It both the train and test data in the format which inDexDa generates.

    Can use:
        - The provided examples with inDexDa
        - New training data provided by user in positive_examples.json and
            negative_examples.json

    :params  N/A
    :return  N/A
    '''
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    # datadir = os.path.join(current_dir, '../data/results.json')

    # Attempts to run the network
    try:
        BertTrain()
    except Exception as bert_error:
        raise Exception(bert_error)
