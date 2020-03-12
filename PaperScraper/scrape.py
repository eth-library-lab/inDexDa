import os
import json
from termcolor import colored

from PaperScraper.utils.command_line import query_yes_no
from PaperScraper.lib.paper_scrape_arxiv import PaperScrapeArXiv
from PaperScraper.lib.paper_scrape_sciencedirect import PaperScrapeScienceDirect


def scrape(archivesToUse, archiveInfo):
    '''
    Will use the web-scraping API to collect academic papers from the specified online
    archive.Saves the scraped papers in a file located in the inDexDa/data folder.

    :params  archivesToUse: list of archives to scrape
             archiveInfo: dict of information retaining to the specific API for the
                            archive
    '''
    current_dir = os.path.dirname(os.path.abspath(__file__))
    for archive in archivesToUse:
        output_dir = os.path.join(current_dir, "data", archive)
        output_file = os.path.join(output_dir, 'papers.json')

        # Make output folder
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        # Query user for web scraping
        question = ('Scan {} for papers?'.format(archive))
        scrape_answer = query_yes_no(question)

        if scrape_answer:
            try:
                scrape_database(archive, archiveInfo, output_file)
            except Exception:
                raise Exception

    # Save Processed Data to New File
    all_papers = []
    for archive in archivesToUse:
        datadir = os.path.join(current_dir, 'data', archive, 'papers.json')

        with open(datadir, 'r') as f:
            contents = f.read()
            papers = json.loads(contents)

        all_papers.extend(papers)

    # Moves the papers from the individual archive data folders to the inDexDa/data
    #  folder. Removes original saved individual archive data files.
    new_dataset_output = os.path.join(current_dir, '../data', 'results.json')
    with open(new_dataset_output, 'w') as f:
        json.dump(all_papers, f, indent=4)

    for archive in archivesToUse:
        datadir = os.path.join(current_dir, 'data', archive, 'papers.json')
        os.remove(datadir)


def scrape_database(archiveToUse, archiveInfo, output_file):
    '''
    Scrapes a selected database for academic papers relating to a query term. Saves
        scraped papers into PaperScraper/data/archive folder.

    :param archiveToUse (string): archive name to scrape
           archiveInfo (list of namedtuples): scraping info for all archives
           output_file: string for output file location
    '''
    databases = {'arxiv': PaperScrapeArXiv,
                 'sciencedirect': PaperScrapeScienceDirect}

    # From the given list of archives to use, selects the appropriate scraper API python
    #   function located in PaperScraper/lib
    config = [archive for archive in archiveInfo if archiveToUse in archive[0]]

    try:
        scraper = databases[archiveToUse.lower()](config[0])
    except Exception:
        raise Exception

    try:
        with open(output_file, 'w') as f:
            json.dump(scraper.papers, f, indent=4)
    except TypeError:
        error = ("Was not able to write to json file.")
        print(colored(error, 'red'))
