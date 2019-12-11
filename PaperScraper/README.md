# Web Scraping
## Setpu With Supported Archives

This portion of the project uses the web-scraping section of the args.json config file.
The two portions of this section are the archives the user wants to scrape for papers and
information on all the archives supported by inDexDa.

### Archives To Use
The user can specify either one or multiple online repositories to scrape from by modifying
the number of entries in the archive tag. The syntax is as follows:
```shell
{"id": "0x", "archive": "name"}
    # x should be an integer between 1-9
    # name is the name of the online paper repository, all lowercase, no spaces
```

### Archive Information
This section is for all the required information for each repository's API. The fields
must be standardized for all archives, so if fields are not needed for a repository
leave them blank.

* All archives require a search query to find papers relating to that term.
* ScienceDirect requires an API key the user must register for themselves (see below) as
well as a range of years to search over.
* Other added archives may require more information, so fields may need to be added and the
scraping code modified.
* Queries for arXiv should only be a single word

* ScienceDirect API Key Application: https://dev.elsevier.com/apikey/manage


## Use New Archive

inDexDa also allows users to use online or local repositories which are not natively
supported. To do this, the following must be done:

1. Create a scraper class in the _PaperScraper/lib/_ folder.
    1. Name of the file with the scraper class should be paper_scrape_name.py where name
       is the name of the archive with no spaces or punctuation between words, all
       letters lowercase.
    2. Class named PaperScraperName where Name is the name of the repository.
    3. Should output a papers.json file which contains a list of dicts, each dict containing
        the title, abstract, authors, category (if available), date of publication (if
        available) of a paper. The papers.json file should be saved to the
        _PaperScraper/data/archiveName_ folder.
    4. Upon being initialized, the new scraper class should scrape the repository and
        compile the list of dicts for the papers. This list should then be set to the
        class variable _self.papers_.
2. From the new file import the class (PaperScraperName) into scrape.py
3. In scrape.py, the databases variable in scrape_databases function needs to be updated
    to include the new scraper. Add a dictionary entry with the key as the name of the
    repository (all lowercase, no spaces or punctuation) and the value as the name of
    the scraping class.


* MAKE SURE _data/sciencedirect/paperinfo.json_ IS DELETED BEFORE RUNNING FOR THE FIRST
TIME WITH A NEW SEARCH QUERY