from lib.paper_scrape import PaperScrape
from lib.paper_scrape_sciencedirect import PaperScrapeScienceDirect
import os
import sys
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--database', type=str, default='arXiv', help='database used')
parser.add_argument('--field', type=str, default='Computer Science',
                    help='field of research')
options = parser.parse_args()

CONFIG_DIR = os.path.join('config', options.database, 'config.json')
OUTPUT_DIR = os.path.join('data', options.database, options.field.replace(' ', ''))
OUTPUT_FILE = os.path.join('data', options.database, options.field.replace(' ', ''),
                           'abstract_links.txt')

if os.path.exists(CONFIG_DIR):
    with open(CONFIG_DIR) as config_json:
        config = json.load(config_json)
    print('Using {} database'.format(options.database))
else:
    print('No config file found at {}'.format(CONFIG_DIR), file=sys.stderr)

if options.database == 'arXiv':
    arXiv = PaperScrape(config)

    if os.path.exists(OUTPUT_DIR):
        with open(OUTPUT_FILE, 'w') as f:
            for item in arXiv.abstract_links:
                f.write(item + '\n')
    else:
        os.mkdir(OUTPUT_DIR)
        with open(OUTPUT_FILE, 'w') as f:
            for item in arXiv.abstract_links:
                f.write(item + '\n')

elif options.database == 'sciencedirect':
    sciencedirect = PaperScrapeScienceDirect(config)

# arXiv.download_papers()
