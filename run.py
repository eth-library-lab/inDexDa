import os
import argparse
from termcolor import colored

from PaperScraper.scrape import scrape
from NLP.train_network import trainNetwork
from NLP.utils.preprocess import PreprocessForBert
from NLP.run_through_network import runThroughNetwork
from DatasetIndexing.infoExtraction import datasetIndexing
from utils import getInfoAboutArchivesToScrape, getInfoAboutNetworkParams


parser = argparse.ArgumentParser()
parser.add_argument('--first_time', dest='first_time', action='store_true')
parser.add_argument('--scrape', dest='scrape', action='store_true')
parser.add_argument('--train', dest='train', action='store_true')
parser.add_argument('--ouput_dir', type=str, default='', help='optional output dir')
parser.set_defaults(first_time=False, scrape=False, train=False)
opt = parser.parse_args()


# Read args.json
try:
    archivesToUse, archiveInfo = getInfoAboutArchivesToScrape()
    networkParams = getInfoAboutNetworkParams()
except Exception:
    output_msg = ('Not able to get information from args.json. Make sure file is formatted'
                 ' correctly. Refer to original format of file provided on GitHub.')
    print(colored(output_msg, 'red'))

if opt.first_time:
    '''
    If it's the first time using inDexDa, certain functions need to be run.
    This script ensures the training examples used for BERT have not been deleted and
    are located in the inDexDa/data folder. It also sets up the required data
    directories for BERT to run and trains the network.

    '''
    output_msg = 'Formatting dataset for BERT network training ...'
    print(colored(output_msg, 'cyan'))

    current_dir = os.path.dirname(os.path.abspath(__file__))
    datadir_pos = os.path.join(current_dir, 'data', 'positive_samples.json')
    datadir_neg = os.path.join(current_dir, 'data', 'negative_samples.json')
    if not os.path.exists(datadir_pos) or not os.path.exists(datadir_neg):
        error = "One or both of the dataset example files is missing."
        fix = ("Make sure inDexDa/data directory exists and that it contains the files "
               " negative_samples.json and positive_samples.json.")
        print(colored(error, 'red'))
        print(colored(fix, 'yellow'))

    # Relevent data preprocessing for BERT network
    preprocess = PreprocessForBert()
    preprocess.processForTrainingBert()

    # Make archive data folders
    datapath = os.path.join(current_dir, 'PaperScraper/data')
    if not os.path.exists(datapath):
        os.mkdir(datapath)

    for archive in archiveInfo:
        archive_datapath = os.path.join(datapath, archive.name)
        if not os.path.exists(archive_datapath):
            os.mkdir(archive_datapath)

    # Train network and save the model
    output_msg = ('Training BERT network for classification of academic papers. This may'
                   ' take awhile ...')
    print(colored(output_msg, 'cyan'))
    trainNetwork()

if opt.scrape and not opt.train:
    '''
    If the user wishes to find new datasets from scraped papers, this section will scrape
        said papers and pass their abstracts to BERT to classify them. Afterwards a file
        will be created which specifies which papers pointed towards a dataset and another
        file containing the specifics of that dataset.
    '''
    # SCRAPE
    output_msg = "Beginning scraping archives for papers ..."
    print(colored(output_msg, 'cyan'))
    try:
        scrape(archivesToUse, archiveInfo)
    except Exception:
        exit()

    # RUN THROUGH NETWORK
    output_msg = "Processing acquired papers through the networks ..."
    print(colored(output_msg, 'cyan'))
    try:
        runThroughNetwork(networkParams)
    except Exception:
        exit()

    # INDEX DATASETS
    output_msg = "Indexing datasets and acquiring more information ..."
    print(colored(output_msg, 'cyan'))
    try:
        datasetIndexing()
    except Exception:
        exit()

if opt.train and not opt.scrape:
    '''
    If the user wishes to find re-train BERT, this script is used. It will preprocess the
        positive and negative training examples into the proper directories, train BERT
        on a portion of them, and then test BERT's accuracy using the remainder of the
        examples.
    '''
    output_msg = "Training the BERT network now ..."
    print(colored(output_msg, 'cyan'))
    trainNetwork(networkParams)

if opt.scrape and opt.train:
    error = ("User specified both --scrape and --train flags. Can only use one of these"
               " flags at a time.")
    print(colored(error, 'red'))
