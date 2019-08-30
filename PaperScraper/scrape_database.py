from lib.paper_scrape import PaperScrape
from lib.paper_scrape_sciencedirect import PaperScrapeScienceDirect
from utils.command_line import query_yes_no

import os
import json


def scrape_database(config, options, CONFIG_DIR, OUTPUT_DIR, OUTPUT_FILE):
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    if options.database == 'arXiv':
        arXiv = PaperScrape(config)

        with open(OUTPUT_FILE, 'w') as f:
            for item in arXiv.abstract_links:
                f.write(item + '\n')


    elif options.database == 'sciencedirect':
        sciencedirect = PaperScrapeScienceDirect(config)

        with open(OUTPUT_FILE, 'w') as f:
            json.dump(sciencedirect.papers, f, indent=4)
