import os
# import json
import argparse
from termcolor import colored

from PaperScraper.scrape import scrape
from NLP.train_network import trainNetwork
from NLP.utils.preprocess import PreprocessForBert
from NLP.run_through_network import runThroughNetwork
from utils import getInfoAboutArchivesToScrape


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
except Exception:
    print('Not able to get information from args.json. Make sure file is formatted'
          ' correctly.')

if opt.first_time:
    '''
    If it's the first time using inDexDa, certain functions need to be run.
    This script ensures the training examples used for BERT have not been deleted and
    are located in the inDexDa/data folder. It also sets up the required data
    directories for BERT to run and trains the network.

    '''
    current_dir = os.path.dirname(os.path.abspath(__file__))
    datadir_pos = os.path.join(current_dir, 'data', 'positive_samples.json')
    datadir_neg = os.path.join(current_dir, 'data', 'negative_samples.json')
    if not os.path.exists(datadir_pos) or not os.path.exists(datadir_neg):
        # error = "Directory with dataset is either empty or does not exist."
        # fix = ("Make sure inDexDa/data directory exists and that it contains the file "
        #        " dataset.json.")
        # print(colored(error, 'red'))
        # print(colored(fix, 'yellow'))

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
    print('Training BERT network for classification of academic papers ...')
    trainNetwork()

if opt.scrape and not opt.train:
    '''
    If the user wishes to find new datasets from scraped papers, this section will scrape
        said papers and pass their abstracts to BERT to classify them. Afterwards a file
        will be created which specifies which papers pointed towards a dataset and another
        file containing the specifics of that dataset.
    '''
    print("Beginning scraping archives for papers ...")
    try:
        scrape(archivesToUse, archiveInfo)
    except Exception as scrape_error:
        print(colored(scrape_error, 'red'))
        fix = ("Archives_to_scrape parameter specified in args.json does not have an "
               " associated scraper class. Make sure to specify either the existing "
               " supported archives (arxiv, sciencedirect) or if new scraper class was "
               " created make sure to update the available databases in "
               " PaperScraper.scrape.scrape_database function.")
        print(colored(fix, 'yellow'))
        exit()

    print("Processing acquired papers through the networks ...")
    try:
        runThroughNetwork()
    except Exception as network_error:
        # print(colored(network_error, 'red'))
        # fix = ("Either specify only existing supported networks in the args.json file or"
        #        " update the available networks in the NLP.run_through_network function.")
        # print(colored(fix, 'yellow'))
        exit()

if opt.train and not opt.scrape:
    '''
    If the user wishes to find re-train BERT, this script is used. It will preprocess the
        positive and negative training examples into the proper directories, train BERT
        on a portion of them, and then test BERT's accuracy using the remainder of the
        examples.
    '''
    print("Training the networks now ...")
    trainNetwork()
