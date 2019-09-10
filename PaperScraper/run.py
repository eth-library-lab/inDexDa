from scrape_database import scrape_database
from papers2mongodb import compile_database
from utils.command_line import query_yes_no

import os
import sys
import json
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--database', type=str, default='arxiv', help='database used')
options = parser.parse_args()

CONFIG_DIR = os.path.join('config', options.database, 'config.json')
OUTPUT_DIR = os.path.join('data', options.database)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'papers.json')

# Make output folder
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

# Check if config file exists for specified database
if os.path.exists(CONFIG_DIR):
    with open(CONFIG_DIR) as config_json:
        config = json.load(config_json)
    print('Using {} database'.format(options.database))
else:
    print('No config file found at {}'.format(CONFIG_DIR), file=sys.stderr)

'''
#########################################################################################
########################### Query user for web scraping #################################
#########################################################################################
'''

question = ('Scan {} for papers?'.format(options.database))
scrape = query_yes_no(question)

if scrape:
    scrape_database(config, options, CONFIG_DIR, OUTPUT_DIR, OUTPUT_FILE)

'''
#########################################################################################
########################### Query user for database update ##############################
#########################################################################################
'''

question = ('Add new papers from {} to database?'.format(options.database))
append = query_yes_no(question)

if append:
    compile_database(options, OUTPUT_FILE)
