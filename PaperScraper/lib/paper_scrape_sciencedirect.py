import os
import json
import math
import time
import requests
import numpy as np
import utils.url_util as urlutil
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
        self.search = config['search']
        self.papers = self.scrape4papers()


    def scrape4papers(self):
        papers = []
        date_range = self.get_dates()
        issue_range = np.arange(1, 100)

        for year in np.nditer(date_range):
            stop_counter = 0    # If 3 consecutive issues have 0 results, go to next year
            for issue in np.nditer(issue_range):
                if stop_counter == 3:
                    break

                data = self.SD_search(year, issue)

                if data == 0:
                    stop_counter += 1
                else:
                    for page in data:
                        for result in page["results"]:
                            try:
                                authors = [d['name'] for d in result["authors"]]
                            except KeyError:
                                authors = []
                            try:
                                doi = result["doi"]
                            except KeyError:
                                doi = []
                            try:
                                title = result["title"]
                            except KeyError:
                                title = []

                            papers.append({"title": title,
                                           "authors": authors,
                                           "doi": doi})
        return papers


    def get_dates(self):
        start_year = int(self.config['start_year'])
        end_year = int(self.config['end_year'])

        if start_year == end_year:
            return start_year
        else:
            return np.arange(start_year, end_year)


    def SD_search(self, date, issue):
        """
        Uses ScienceDirect Search API v2 to find possible matches over the entire database.
        The API mandates a limit of 3 queries per second, so a delay must be used.

        :params  date: year to search over (string)
                 issue: issue number to search over (string)
        :return
        """

        results = []

        while True:
            try:
                data = self.APIRequest(date, issue, 0)
                time.sleep(.5)
                totalResults = data['resultsFound']
                break
            except json.JSONDecodeError:
                time.sleep(1)

        if totalResults > 0:
            for pagenum in range(math.ceil(totalResults/100)):
                progress.printProgressBar(pagenum + 1, math.ceil(totalResults/100),
                                          prefix='Progress [Year:{}, Issue:{}]:'.format(date, issue),
                                          suffix='Complete', length=30)
                start_idx = pagenum * 100

                while True:
                    try:
                        results.append(self.APIRequest(date, issue, start_idx))
                        time.sleep(.5)
                        break
                    except json.JSONDecodeError:
                        time.sleep(1)

            return results
        else:
            return 0


    def APIRequest(self, date, issue, start_idx):

        if not isinstance(date, str):
            date = str(date)
        if not isinstance(issue, str):
            issue = str(issue)
        if not isinstance(start_idx, str):
            start_idx = str(start_idx)

        key = self.apikey
        search = self.search
        url = 'https://api.elsevier.com/content/search/sciencedirect'

        headers = {"Accept": "application/json",
                   "X-ELS-APIKey": key,
                   "content-type": "application/json"}

        body = json.dumps({"qs": search,
                           "display": {
                                "offset": start_idx,
                                "show": "100",
                                "sortBy": "relevance"
                                      },
                           "date": date,
                           "openAccess": "true",
                           "issue": issue})

        r = requests.put(url, headers=headers, data=body)
        return json.loads(str(r.text))

