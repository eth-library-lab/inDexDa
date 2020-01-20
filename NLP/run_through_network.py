import os
from NLP.BERT.lib.test import test as BertTest


def runThroughNetwork(networkParams):
    '''
    Takes scraped papers and runs them through the pre-trained BERT classification
    network. Takes the original scraped info from the papers modifies the file such
    that:
        - Only papers which indicate a newly created dataset are kept
        - Each dict is appended with a "Predicted": "Dataset Detected" field

    Specific actions need to be taken when using the ktrain BERT network. For more
    information please refer to the inDexDa manual.

    :params  networkParams: [epochs, bathSize]
    :return  N/A
    '''
    current_dir = os.path.dirname(os.path.abspath(__file__))
    datadir = os.path.join(current_dir, '../data/results.json')

    # Attempts to run the network
    try:
        BertTest(datadir, networkParams[1])
    except Exception as bert_error:
        raise Exception(bert_error)
