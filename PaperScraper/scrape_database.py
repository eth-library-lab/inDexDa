import json

from lib.paper_scrape_arxiv import PaperScrapeArXiv
from lib.paper_scrape_sciencedirect import PaperScrapeScienceDirect


def scrape_database(config, options, CONFIG_DIR, OUTPUT_DIR, OUTPUT_FILE):
    databases = {'arxiv': PaperScrapeArXiv,
                 'sciencedirect': PaperScrapeScienceDirect}

    scraper = databases[options.database.lower()](config)

    with open(OUTPUT_FILE, 'w') as f:
            json.dump(scraper.papers, f, indent=4)
