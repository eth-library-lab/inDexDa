import os

from PaperScraper.lib.paper_scrape_arxiv import PaperScrapeArXiv
from PaperScraper.lib.paper_scrape_sciencedirect import PaperScrapeScienceDirect


class ExtractInfoFromPaper():
    def __init__(self, paper):
        self.paper = paper

    def getPDF(self):

        archive = paper["Archive"]
        databases = {'arxiv': ArXivScraper,
                     'sciencedirect': ScienceDirectScraper}

        try:
            scraper = databases[archive.lower()](config[0])
