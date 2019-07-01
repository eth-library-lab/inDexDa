import os
import json
import argparse
from lib.paper_scrape import PaperScrape

parser = argparse.ArgumentParser()
parser.add_argument('--database', type=str, default='arXiv', help='database used')
parser.add_argument('--field', type=str, default=12, help='field of research')

options = parser.parse_args()
print (options)

CONFIG_PATH = os.path.join('config' + options.database + 'config.json')

with open('config/arXiv/config.json') as config_json:
    config = json.load(config_json)

arXiv = PaperScrape(config)

with open('data/arXiv/pdf_links.txt', 'w') as f:
    for item in arXiv.pdf_links:
        f.write(item + '\n')

# arXiv.download_papers()
