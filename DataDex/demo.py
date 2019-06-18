import json
from get_papers import get_link

with open('config/arXiv/config.json') as config_json:
    config = json.load(config_json)

get_link(config)
