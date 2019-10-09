import json
# import time
# import utils.command_line as progress

from utils.xploreapi import XPLORE


class PaperScrapeIEEEXplore:
    def __init__(self, config):
        """
        Create an for storing links to papers in a given topic

        :param config: config.json file containing information about which
                        topic papers will be found in
        """

        self.config = config
        self.apikey = config['apiKey']
        self.search = config['search']
        self.papers = self.scrape4papers()

    def scrape4papers(self):
        print('TO DO')

    def APIRequest(self, start_idx):
        '''
        IEEEXplore API use requires a valid API key which you can register at:
            https://developer.ieee.org/member/register


        :params   start_idx: skip to this result (int)
        :return   (dict) contains search results for given page
        '''
        xplore = XPLORE(self.apikey)

        xplore.maximumResults(200)
        xplore.indexTerms('dataset')
        xplore.queryText('dataset')

        results = json.loads(str(xplore.callAPI()))

        if results is not {}:
            return results
        else:
            return None
