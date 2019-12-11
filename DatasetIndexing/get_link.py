import os
import json


def getLink():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, '../data/results.json')

    with open(data_dir, 'r') as f:
        contents = f.read()
        papers = json.loads(contents)
