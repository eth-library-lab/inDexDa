from utils import getInfoAboutArchivesToScrape
from PaperScraper.utils.command_line import printProgressBar
from DatasetIndexing.lib.arxiv_scraper import ArXivScraper
from DatasetIndexing.lib.sciencedirect_scraper import ScienceDirectScraper

import os
import json
from rake_nltk import Rake
from termcolor import colored
from tqdm.contrib.concurrent import process_map



def datasetIndexing():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    datadir = os.path.join(current_dir, '../data', 'results.json')

    try:
        with open(datadir, 'r') as f:
            contents = f.read()
            papers = json.loads(contents)

    except:
        error = "Was not able to read results.json file in datasetIndexing."
        print(colored(error, 'red'))

    papers = process_map(multiprocessScrape, papers, max_workers=10)

    # new_papers = []
    # num_papers = len(papers)
    # for idx, paper in enumerate(papers):
    #     printProgressBar(idx + 1, num_papers, prefix='Progress :', suffix='Complete',
    #                      length=30)
    #     extractor = ExtractInfoFromPaper(paper)
    #     new_paper = extractor.extract()

    #     new_papers.append(new_paper)

    outputdir = os.path.join(current_dir, '../data/final_results.json')
    with open(outputdir, 'w') as f:
        json.dump(papers, f, indent=4)

#=========================================================================================
#=========================================================================================
def multiprocessScrape(paper):
    '''
    We can use multiprocessing to retrieve paper's body text for analysis to
      significantly reduce the time requirements of scraping.

    :params  paper: dictionary of info about a paper
    :return  paper: dictionary of updated info about the paper
    '''
    archives, archive_info = getInfoAboutArchivesToScrape()
    config = [paper, archive_info]

    archive = paper["Archive"]
    databases = {'arxiv': ArXivScraper,
                 'sciencedirect': ScienceDirectScraper}

    try:
        scraper = databases[archive.lower()](config)
        return scraper.extract()
        print('WAIT')
    except Exception as e:
        return None
        # error = "\nExtraction failed"
        # print(colored(error, 'red'))
        # print(colored(e, 'yellow'))
        # raise Exception

#=========================================================================================
#=========================================================================================


# class ExtractInfoFromPaper():
#     def __init__(self, paper):
#         self.paper = paper
#         archives, archive_info = getInfoAboutArchivesToScrape()
#         self.config = [paper, archive_info]

#     def extract(self):
#         archive = self.paper["Archive"]
#         databases = {'arxiv': ArXivScraper,
#                      'sciencedirect': ScienceDirectScraper}

#         try:
#             scraper = databases[archive.lower()](self.config)
#             return scraper.extract()
#         except Exception as e:
#             error = "\nExtraction failed"
#             print(colored(error, 'red'))
#             print(colored(e, 'yellow'))
#             raise Exception


# if __name__ == '__main__':
#     paper = {
#         "Title": "BomJi at SemEval-2018 Task 10: Combining Vector-, Pattern- and  Graph-based Information to Identify Discriminative Attributes",
#         "Abstract": "This paper describes BomJi, a supervised system for capturing discriminative attributes in word pairs (e.g. yellow as discriminative for banana over watermelon). The system relies on an XGB classifier trained on carefully engineered graph-, pattern- and word embedding based features. It participated in the SemEval- 2018 Task 10 on Capturing Discriminative Attributes, achieving an F1 score of 0:73 and ranking 2nd out of 26 participant systems.",
#         "Authors": [
#             "Enrico Santus",
#             "Chris Biemann",
#             "Emmanuele Chersoni"
#         ],
#         "Date": "2018-04-30T14:58:22Z",
#         "DOI": [],
#         "Category": [
#             "cs.CL"
#         ],
#         "Link": "http://arxiv.org/pdf/1804.11251v1.pdf",
#         "Archive": "arXiv",
#         "Prediction": "Dataset Detected"}
#     test = ExtractInfoFromPaper(paper)
#     test.extract()
