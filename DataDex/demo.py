import json
from lib.paper_scrape import PaperScrape

with open('config/arXiv/config.json') as config_json:
    config = json.load(config_json)

arXiv = PaperScrape(config)

with open('data/arXiv/pdf_links.txt', 'w') as f:
    for item in arXiv.pdf_links:
        f.write(item + '\n')

# arXiv.download_papers()
