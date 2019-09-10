import json
import math
import time
import requests
import numpy as np
import utils.command_line as progress


class PaperScrapeScienceDirect:
    def __init__(self, config):
        """
        Create an for storing links to papers in a given topic

        :param config: config.json file containing information about which
                        topic papers will be found in
        """

        self.config = config
        self.apikey = config['apiKey']
        self.query = config['query']
        self.papers = self.scrape4papers()

    def scrape4papers(self):
        '''
        Searches through ScienceDirect archive for papers relating to the search term and
        date ranges specified within the ScienceDirect config file.
        '''
        papers = []
        date_range = self.get_dates()
        issue_range = np.arange(1, 100)
        # issue_range = np.arange(1, 5)

        # Uses ScienceDirect Search V2 to do a general search across the query params
        print('Scraping papers from ScienceDirect using query term "%s" :' % (self.query))

        for year in np.nditer(date_range):
            stop_counter = 0    # If 3 consecutive issues have 0 results, go to next year
            for issue in np.nditer(issue_range):
                if stop_counter == 3:
                    break

                data = self.ScienceDirectSearchV2(year, issue)

                if data == 0:
                    stop_counter += 1
                else:
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
                                           "DOI": doi})

        # Remove any papers without DOIs
        papers[:] = [d for d in papers if d.get('DOI') != []]

        # More refined search using Aabstract Retrieval API to add abstract to papers dict
        print('Retrieving abstracts for discovered papers:')

        length = len(papers)
        for i, paper in enumerate(papers):
            progress.printProgressBar(i + 1, length, prefix='Progress: ',
                                      suffix='Complete', length=50)
            request = Request('retrieval', DOI=paper['DOI'])
            article = self.APIRequest(request)

            if article is not None:
                category = []
                try:
                    for subject in article['abstracts-retrieval-response']['subject-areas']['subject-area']:
                        if subject not in category:
                            category.append(subject['@abbrev'])
                except KeyError:
                    category = []

                try:
                    abstract = article['abstracts-retrieval-response']['item']['bibrecord']['head']['abstracts']
                except KeyError:
                    abstract = []

                paper['Absrtact'] = abstract
                paper['Category'] = category
            else:
                paper['Absrtact'] = []
                paper['Category'] = []

        # Remove any papers without Abstracts, Authors, or Titles
        papers[:] = [d for d in papers if d.get('Absrtact') != []]
        papers[:] = [d for d in papers if d.get('Authors') != []]
        papers[:] = [d for d in papers if d.get('Title') != []]

        return papers

    def get_dates(self):
        '''
        Makes array of year range specified within ScienceDirect config file. Robust to
        only one year entry.
        :return numpy array (ints) or int
        '''
        start_year = int(self.config['start_year'])
        end_year = int(self.config['end_year'])

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

        results = []

        while True:
            try:
                request = Request('search', date=date, issue=issue, start_idx=0)

                data = self.APIRequest(request)
                time.sleep(.5)
                totalResults = data['resultsFound']
                break
            except json.JSONDecodeError:
                time.sleep(1)

        if totalResults > 0:
            for pagenum in range(math.ceil(totalResults / 100)):
                prefix = 'Progress [Year:{}, Issue:{}]:'.format(date, issue)
                progress.printProgressBar(pagenum + 1, math.ceil(totalResults / 100),
                                          prefix=prefix, suffix='Complete', length=30)
                start_idx = pagenum * 100

                while True:
                    try:
                        request = Request('search', date=date, issue=issue,
                                          start_idx=start_idx)
                        results.append(self.APIRequest(request))
                        time.sleep(.5)
                        break
                    except json.JSONDecodeError:
                        time.sleep(1)

            return results
        else:
            return 0

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
            return json.loads(str(r.text))

        # ABSTRACT RETRIEVAL
        elif request.request_type == 'retrieval':
            try:
                key = self.apikey
                url = 'https://api.elsevier.com/content/abstract/doi/'
                url = url + request.DOI + '?'
                url = url + 'apiKey=' + key
                url = url + '&httpAccept=application%2Fjson'
            except TypeError:
                return None

            r = requests.get(url)

            try:
                return json.loads(str(r.text))
            except ValueError:
                return None


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
