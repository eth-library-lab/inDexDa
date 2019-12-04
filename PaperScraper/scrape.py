import os
import json

from PaperScraper.papers2mongodb import compile_database
from PaperScraper.utils.command_line import query_yes_no
from PaperScraper.lib.paper_scrape_arxiv import PaperScrapeArXiv
from PaperScraper.lib.paper_scrape_sciencedirect import PaperScrapeScienceDirect


def scrape(archivesToUse, archiveInfo):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    for archive in archivesToUse:
        output_dir = os.path.join(current_dir,
                                  "data",
                                  archive)
        output_file = os.path.join(output_dir, 'papers.json')

        # Make output folder
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        # ========================================================== #
        # ============ Query user for web scraping ================= #
        # ========================================================== #

        question = ('Scan {} for papers?'.format(archive))
        scrape_answer = query_yes_no(question)

        if scrape_answer:
            try:
                scrape_database(archive, archiveInfo, output_file)
            except Exception as database_error:
                raise Exception(database_error)

        # ========================================================== #
        # =========== Save Processed Data to New File ============== #
        # ========================================================== #

        all_papers = []
        for archive in archivesToUse:
            datadir = os.path.join(current_dir, 'data', archive, 'papers.json')

            with open(datadir, 'r') as f:
                contents = f.read()
                papers = json.loads(contents)

            all_papers.extend(papers)

        new_dataset_output = os.path.join(current_dir, '../data', 'samples.json')
        with open(new_dataset_output, 'w') as f:
            json.dump(all_papers, f, indent=4)

        # ========================================================== #
        # =========== Query user for database update =============== #
        # ========================================================== #

        question = ('Add new papers from {} to database?'.format(archive))
        append_database = query_yes_no(question)

        if append_database:
            compile_database(output_file)


def scrape_database(archiveToUse, archiveInfo, output_file):
    '''
    Scrapes a selected database for academic papers relating to a query term. Saves
        scraped papers into PaperScraper/data/archive folder.

    :param archiveToUse (string): archive name to scrape
           archiveInfo (list of namedtuples): scraping info for all archives
    '''
    databases = {'arxiv': PaperScrapeArXiv,
                 'sciencedirect': PaperScrapeScienceDirect}

    config = [archive for archive in archiveInfo if archiveToUse in archive[0]]

    try:
        scraper = databases[archiveToUse.lower()](config[0])
    except Exception as database_error:
        raise Exception('Specified archive does not have a scrape class.')

    with open(output_file, 'w') as f:
        json.dump(scraper.papers, f, indent=4)
