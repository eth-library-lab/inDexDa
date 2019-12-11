import math
import requests
import feedparser
import PaperScraper.utils.command_line as progress

from PaperScraper.utils.url_util import get_url


class PaperScrapeArXiv:
    def __init__(self, config):
        """
        Create an for storing links to papers in a given topic

        :param config: namedtuple containing information about which
                        topic papers will be found in
        :return papers (list of dict): dict of each paper found with relevant content
        """
        self.config = config
        self.query = config.query
        self.papers = self.scrape4papers()

    def scrape4papers(self):
        '''
        Calls the arXiv API to scrape all papers relating to the search term. The API then
        returns the dict of results which we parse here. If one of the fields is not in
        the dict, we leave it as a blank space.

        :params  N/A
        :return  papers: list of dicts, each dict containing info on a specific paper
        '''
        papers = []
        results = self.arXiv_search()

        for page in results:
            for result in page:
                try:
                    authors = [d['name'] for d in result["authors"]]
                    if len(authors) > 3:
                        del authors[3:]
                except KeyError:
                    authors = []
                try:
                    title = result["title"].replace('\n', '')
                except KeyError:
                    title = []
                try:
                    pubdate = result["published"]
                except KeyError:
                    pubdate = []
                try:
                    abstract = result['summary'].replace('\n', ' ')
                except KeyError:
                    abstract = []
                try:
                    doi = result["arxiv_doi"]
                except KeyError:
                    doi = []

                category = []
                try:
                    for tag in result['tags']:
                        category.append(tag['term'])
                except KeyError:
                    category = []

                papers.append({"Title": title,
                               "Abstract": abstract,
                               "Authors": authors,
                               "Date": pubdate,
                               "DOI": doi,
                               "Category": category})

        return papers

    def arXiv_search(self):
        '''
        Determines number of pages to search through, then calls the ArXiv API for each
        page.

        :params  N/A
        :return  results (list of dict): entire parsed json file from the get command
        '''
        start = '0'
        url = self.seturl(start, max_results=1)

        r = requests.get(url)
        feed = feedparser.parse(r.text)
        totalResults = int(feed.feed['opensearch_totalresults'])

        # If we receive no results from a search query, we simply return 0
        results = []
        if totalResults > 0:
            for pagenum in range(math.ceil(totalResults / 100)):
                progress.printProgressBar(pagenum + 1, math.ceil(totalResults / 100),
                                          prefix='Progress :', suffix='Complete',
                                          length=30)
                start_idx = pagenum * 100

                results.append(self.APIRequest(start_idx))

            return results
        else:
            return 0

    def APIRequest(self, start_idx):
        '''
        Gets the url for the specified page, returns contents of that page

        :params  start_idx (int or str): result number to start at
        :return  feed['entries'] (json): parsed contents of webpage
        '''
        url = self.seturl(start_idx)

        r = get_url(url)
        if r is not None:
            feed = feedparser.parse(r.text)
            return feed['entries']
        else:
            return None

    def seturl(self, start_idx, max_results=500):
        '''
        Creates url for requests.get command

        :params  start_idx (int or str): starting index for results
                 max_results (int or str): maximum number of results per page
        :return  url (str): url for webpage
        '''
        if not isinstance(start_idx, str):
            start_idx = str(start_idx)
        if not isinstance(max_results, str):
            max_results = str(max_results)
        query = self.query

        url = 'http://export.arxiv.org/api/query?'
        url = url + 'search_query=all:' + query
        url = url + '&start=' + start_idx
        url = url + '&max_results=' + max_results

        return url


# if __name__ == '__main__':
    # config = ['arxiv', 'data']

    # arxiv = PaperScrapeArXiv(config)
    # paper = arxiv.APIRequest(0)
    # print(paper[0]['arxiv_doi'])
