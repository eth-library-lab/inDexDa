import os
import json

from utils import getInfoAboutArchivesToScrape
from DatasetIndexing.lib.arxiv_scraper import ArXivScraper
from DatasetIndexing.lib.sciencedirect_scraper import ScienceDirectScraper


def datasetIndexing():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    datadir = os.path.join(current_dir, '../data', 'results.json')
    with open(datadir, 'r') as f:
        contents = f.read()
        papers = json.loads(contents)

    for paper in papers:
        extractor = ExtractInfoFromPaper(paper)
        extractor.extract()


class ExtractInfoFromPaper():
    def __init__(self, paper):
        self.paper = paper
        archives, archive_info = getInfoAboutArchivesToScrape()
        self.config = [paper, archive_info]

    def extract(self):

        archive = paper["Archive"]
        databases = {'arxiv': ArXivScraper,
                     'sciencedirect': ScienceDirectScraper}

        try:
            scraper = databases[archive.lower()](self.config)
        except:
            raise Exception


if __name__ == '__main__':
    paper = {
        "Title": "Dataset for Evaluating the Accessibility of the Websites of Selected Latin American Universities",
        "DOI": '10.1016/j.dib.2019.105013',
        "Archive": "sciencedirect",
        "Prediction": "Dataset Detected"}
    test = ExtractInfoFromPaper(paper)
    test.extract()
