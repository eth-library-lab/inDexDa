import os
import json
import requests
import urllib.request
from bs4 import BeautifulSoup


key = '8195584e9a1784037041888ba25292ee'
url = 'https://api.elsevier.com/content/search/sciencedirect'

headers = {"Accept": "application/json",
           "X-ELS-APIKey": key,
           "content-type": "application/json"}

body = {"qs": "dataset",
        # "date": "2018",
        "sortBy": "relevance",
        "show": "100",
        "openAccess": "true",
        "issue": "0"}
body = json.dumps(body)

r = requests.put(url, headers=headers, data=body)

dirname = os.getcwd()
data = json.loads(str(r.text))

file = os.path.join(dirname, 'log/test.json')
with open(file, 'w') as f:
    json.dump(data, f, indent=4)

