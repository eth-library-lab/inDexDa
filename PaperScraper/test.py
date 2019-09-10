import json
import math
import time
import requests
import numpy as np
import utils.command_line as progress

def APIRequest(request):
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
    key = '8195584e9a1784037041888ba25292ee'

    url = 'https://api.elsevier.com/content/abstract/doi/'
    url = url + request.DOI + '?'
    url = url + 'apiKey=' + key
    url = url + '&httpAccept=application%2Fjson'

    r = requests.get(url)
    json_obj = json.loads(str(r.text))

    with open('log/test.json', 'w') as f:
        json.dump(json_obj, f, indent=4)

    category = []
    for subject in json_obj['abstracts-retrieval-response']['subject-areas']['subject-area']:
        category.append(subject['@abbrev'])

    abstract = json_obj['abstracts-retrieval-response']['item']['bibrecord']['head']['abstracts']

    print(abstract)
    print(category)
    input()

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
            request_type= 'search'

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


if __name__ == '__main__':
    request = Request('retrieval', DOI='10.1016/S0034-4257(99)00099-1')
    APIRequest(request)
