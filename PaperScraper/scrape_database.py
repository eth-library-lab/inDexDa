# import os
import json

from lib.paper_scrape_arxiv import PaperScrapeArXiv
from lib.paper_scrape_ieeexplore import PaperScrapeIEEEXplore
from lib.paper_scrape_sciencedirect import PaperScrapeScienceDirect


def scrape_database(config, options, CONFIG_DIR, OUTPUT_DIR, OUTPUT_FILE):
    # if options.database == 'arXiv':
    #     arXiv = PaperScrapeArXiv(config)

    #     with open(OUTPUT_FILE, 'w') as f:
    #         json.dump(arXiv.papers, f, indent=4)

    # elif options.database == 'sciencedirect':
    #     sciencedirect = PaperScrapeScienceDirect(config)

    #     with open(OUTPUT_FILE, 'w') as f:
    #         json.dump(sciencedirect.papers, f, indent=4)

    # elif options.database == 'ieeexplore':
    #     ieeexplore = PaperScrapeIEEEXplore(config)

    #     with open(OUTPUT_FILE, 'w') as f:
    #         json.dump(ieeexplore.papers, f, indent=4)

    databases = {'arxiv': PaperScrapeArXiv,
                 'ieeexplore': PaperScrapeIEEEXplore,
                 'sciencedirect': PaperScrapeScienceDirect}

    scraper = databases[options.database.lower()](config)

    with open(OUTPUT_FILE, 'w') as f:
            json.dump(scraper.papers, f, indent=4)
