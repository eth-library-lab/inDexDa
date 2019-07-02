import os
import json
import argparse
from lib.paper_scrape import PaperScrape

parser = argparse.ArgumentParser()
parser.add_argument('--database', type=str, default='arXiv', help='database used')
parser.add_argument('--field', type=str, default=12, help='field of research')
options = parser.parse_args()


CONFIG_DIR = os.path.join('..' + 'config' + options.database + 'config.json')
OUTPUT_DIR = os.path.join('..' + 'data' + options.field + 'pdf_links.txt')

with open(CONFIG_FILE) as config_json:
    config = json.load(config_json)

arXiv = PaperScrape(config)

with open(OUTPUT_DIR, 'w') as f:
    for item in arXiv.pdf_links:
        f.write(item + '\n')

# arXiv.download_papers()
