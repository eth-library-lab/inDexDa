import os
import json
import math
import time
import requests
import numpy as np

import PaperScraper.utils.command_line as progress


class PaperScrapeScienceDirect:
    def __init__(self, config):
        """
        Create an for storing links to papers in a given topic. ScienceDirect API is quite
        slow, so scraping is done in 2 phases: scrape general information on each paper,
        then scrape the abstract for each paper. Scraping the abstracts can take hours
        or even days, so it sometimes times-out. If this happens, the general scraped
        info is saved to a file.

        ScienceDirect API is complicated, so please refer to the inDexDa manual for more
        information if confused.

        :param config: namedtuple containing information about which
                        topic papers will be found in
        """

        self.config = config
        self.query = config.query
        self.apikey = config.apikey

        # Checks if scrape4abstracts had failed in a previous run
        current_dir = os.path.dirname(os.path.abspath(__file__))
        datapath = os.path.join(current_dir, '../data/sciencedirect/paperinfo.json')

        # If general scraped info was saved previously (meaning abstract scraper failed)
        if os.path.exists(datapath):
            with open(datapath, 'r') as f:
                contents = f.read()
                papers = json.loads(contents)
            self.papers = self.scrape4abstracts(papers, datapath)
        else:
            self.papers = self.scrape4papers()
            self.savePaperInformation(self.papers)
            self.papers = self.scrape4abstracts(self.papers, datapath)

    def scrape4papers(self):
        '''
        Searches through ScienceDirect archive for papers relating to the search term and
        date ranges specified within the ScienceDirect config file.

        :params  N/A
        :return  papers: list of dicts, each dict containing info on a specific paper
        '''
        papers = []
        date_range = self.getDates()
        issue_range = np.arange(1, 100)

        # Uses ScienceDirect Search V2 to do a general search across the query params
        print('Scraping papers from ScienceDirect using query term "%s" :' % (self.query))

        for year in np.nditer(date_range):
            stop_counter = 0    # If 3 consecutive issues have 0 results, go to next year
            for issue in np.nditer(issue_range):
                if stop_counter == 3:
                    break

                # Calls API
                data = self.ScienceDirectSearchV2(year, issue)

                if data == 0:
                    stop_counter += 1
                else:
                    # Parse information from each dict
                    for page in data:
                        for result in page["results"]:
                            try:
                                authors = [d['name'] for d in result["authors"]]
                                if isinstance(authors, list) and len(authors) > 3:
                                    del authors[3:]
                            except KeyError:
                                authors = []
                            try:
                                doi = result["doi"]
                                if isinstance(doi, list) and len(doi) > 1:
                                    del doi[1:]
                            except KeyError:
                                doi = []
                            try:
                                title = result["title"]
                                if isinstance(title, list) and len(title) > 1:
                                    del title[1:]
                            except KeyError:
                                title = []
                            try:
                                pubdate = result["publicationDate"]
                            except KeyError:
                                pubdate = []

                            papers.append({"Title": title,
                                           "Authors": authors,
                                           "Date": pubdate,
                                           "DOI": doi,
                                           "Category": ""})

        # Remove any papers without DOIs as we will be unable to find their abstracts
        papers[:] = [d for d in papers if d.get('DOI') != []]

        print("Finished getting basic info on papers")

        return papers

    def scrape4abstracts(self, papers, datapath):
        '''
        More refined search using Abstract Retrieval API to add abstract to papers dict.

        :params  papers: list of dicts, each dict with info on a specific paper
                 datapath: path to previously saved general scraped data
        '''
        print('Retrieving abstracts for discovered papers:\n')

        length = len(papers)
        for i, paper in enumerate(papers):
            progress.printProgressBar(i + 1, length, prefix='Progress: ',
                                      suffix='Complete', length=50)
            request = Request('retrieval', DOI=paper['DOI'])

            try:
                # Call API
                article = self.APIRequest(request)
            except Exception as error:
                print(error)
                continue

            if article is not None:
                try:
                    abstract = article['full-text-retrieval-response']['coredata']['dc:description']
                    if "Abstract" in abstract:
                        abstract = abstract.replace('Abstract ', '')
                    if "Background" in abstract:
                        abstract = abstract.replace('Background ', '')
                except (KeyError, TypeError):
                    abstract = []

                paper['Abstract'] = abstract
            else:
                paper['Abstract'] = []
                paper['Category'] = []

        # Remove any papers without Abstracts, Authors, or Titles
        papers[:] = [d for d in papers if d.get('Abstract') != []]
        papers[:] = [d for d in papers if d.get('Abstract') is not None]
        papers[:] = [d for d in papers if d.get('Authors') != []]
        papers[:] = [d for d in papers if d.get('Title') != []]

        # Remove file containing general scraped data
        os.remove(datapath)

        return papers

    def savePaperInformation(self, papers):
        '''
        After scraping general info from each paper, save this in a json file so that
        if the abstract retreival fails we can load this file and skip the general
        scraping next attempt.

        :params  papers: list of dicts, each dict containing info on a specific paper
        '''
        print("Saving basic info on papers")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.join(current_dir, '../data/sciencedirect/paperinfo.json')

        with open(save_dir, 'w') as f:
            json.dump(papers, f, indent=4)

    def getDates(self):
        '''
        Makes array of year range specified within args config file. Robust to
        only one year entry.

        :params  N/A
        :return  numpy array (ints) or int
        '''
        start_year = int(self.config.start_year)
        end_year = int(self.config.end_year)

        if start_year == end_year:
            return start_year
        else:
            return np.arange(start_year, end_year + 1)

    def ScienceDirectSearchV2(self, date, issue):
        """
        Uses ScienceDirect Search API v2 to find possible matches over the entire
        database. The API mandates a limit of 3 queries per second, so a delay must be
        used.

        :params  date: year to search over (string)
                 issue: issue number to search over (string)
        :return  results: json object for page of ScienceDirect API search results
        """

        try:
            results = []

            request = Request('search', date=date, issue=issue, start_idx=0)

            data = self.APIRequest(request)
            totalResults = data['resultsFound']

            if totalResults > 0:
                for pagenum in range(math.ceil(totalResults / 100)):
                    prefix = 'Progress [Year:{}, Issue:{}]:'.format(date, issue)
                    progress.printProgressBar(pagenum + 1, math.ceil(totalResults / 100),
                                              prefix=prefix, suffix='Complete', length=30)
                    start_idx = pagenum * 100

                    request = Request('search', date=date, issue=issue, start_idx=start_idx)
                    results.append(self.APIRequest(request))

                return results
            else:
                return 0
        except Exception as error:
            raise Exception(error)

    def APIRequest(self, request):
        '''
        Elsevier API handler for ScienceDirect. Supports both ScienceDirect Search V2 and
        Abstract Retrieval APIs (based on request_type param).

        Search V2: Uses supported PUT method with date, issue, and start_idx params.
        Abstract Retrieval: Uses GET method with DOI param.

        Headers need to include active API Key provided by the Elsevier website.
        https://dev.elsevier.com/apikey/manage

        :params   date: year for search (int)
                  issue: issue number for search (int)
                  start_idx: skip to this result (int)
        :return   (dict) contains search results for given page
        '''

        # SCIENCEDIRECT SEARCH V2
        while True:
            # Will try to access a website 5 times before exiting the function. Sometimes
            #  a bad response occurs so we test multiple times to ensure something failed.
            counter = 0
            try:
                # GENERAL SEARCH
                if request.request_type == 'search':
                    key = self.apikey
                    query = self.query
                    url = 'https://api.elsevier.com/content/search/sciencedirect'

                    headers = {"Accept": "application/json",
                               "X-ELS-APIKey": key,
                               "content-type": "application/json"}

                    body = json.dumps({"qs": query,
                                       "display": {"offset": request.start_idx,
                                                   "show": "100",
                                                   "sortBy": "relevance"},
                                       "date": request.date,
                                       "openAccess": "true",
                                       "issue": request.issue})

                    r = requests.put(url, headers=headers, data=body)
                    time.sleep(.5)
                    return json.loads(str(r.text))

                # ABSTRACT RETRIEVAL
                elif request.request_type == 'retrieval':
                    key = self.apikey

                    url = 'https://api.elsevier.com/content/article/doi/'
                    url = url + request.DOI

                    headers = {"Accept": "application/json",
                               "X-ELS-APIKey": key,
                               "content-type": "application/json"}

                    r = requests.get(url, headers=headers)
                    time.sleep(.1)

                    return json.loads(str(r.text))

            except:
                # If query failed, wait for 1 second and try again
                counter += 1
                time.sleep(1)

                # If enough failed attempts occur, exit function
                if counter == 5:
                    raise Exception("ScienceDirect has stopped responding.")


class Request():
    def __init__(self, request_type, date=None, issue=None, start_idx=None, DOI=None):
        '''
        Class containing parameters for API request. Supports both ScienceDirect Search
        V2 and Abstract Retieval API requests.

        :params  type (str) : type of request ('search' or 'retrieval')
                 date (str) : year for Search V2
                 issue (str): issue number for Search V2
                 DOI (str)  : DOI number of Abstract Retrieval
        '''

        if request_type != 'search' and request_type != 'retrieval':
            request_type = 'search'

        if DOI and not isinstance(DOI, str):
            DOI = str(DOI)
        if date and not isinstance(date, str):
            date = str(date)
        if issue and not isinstance(issue, str):
            issue = str(issue)
        if start_idx and not isinstance(start_idx, str):
            start_idx = str(start_idx)

        self.request_type = request_type
        self.DOI = DOI
        self.date = date
        self.issue = issue
        self.start_idx = start_idx


# if __name__ == '__main__':
    # results = []
    # config = ['sciencedirect', 'dataset', '8195584e9a1784037041888ba25292ee']

    # scrape = PaperScrapeScienceDirect(config)

    # request = Request('retrieval',
    #                   date=2016,
    #                   issue=1,
    #                   start_idx=0,
    #                   DOI='10.1016/S0378-3774(99)00085-2')

    # article = scrape.APIRequest(request)
    # abstract = article['full-text-retrieval-response']['coredata']['dc:description']
    # print(abstract.replace('Abstract ', ''))
