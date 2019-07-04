import os
import sys
import json
import argparse
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from lib.paper_scrape import PaperScrape


parser = argparse.ArgumentParser()
parser.add_argument('--database', type=str, default='arXiv', help='database used')
parser.add_argument('--field', type=str, default='Computer Science',
                    help='field of research')
options = parser.parse_args()

CONFIG_DIR = os.path.join('../' + 'config/' + options.database + '/config.json')
OUTPUT_DIR = os.path.join('../' + 'data/' + options.field + '/pdf_links.txt')

with open(CONFIG_DIR) as config_json:
    config = json.load(config_json)

arXiv = PaperScrape(config)

with open(OUTPUT_DIR, 'w') as f:
    for item in arXiv.pdf_links:
        f.write(item + '\n')

# arXiv.download_papers()
