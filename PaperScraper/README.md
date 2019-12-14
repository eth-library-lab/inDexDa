<div align="center">
  <a href="https://www.librarylab.ethz.ch"><img src="https://www.librarylab.ethz.ch/wp-content/uploads/2018/05/logo.svg" alt="ETH Library LAB logo" height="160"></a>
  
  <br/>
  
  <p><strong>inDexDa</strong> - Natural Language Processing of academic papers for dataset identification and indexing.</p>
  
  <p>An Initiative for human-centered Innovation in the Knowledge Sphere of the <a href="https://www.librarylab.ethz.ch">ETH Library Lab</a>.</p>

</div>

## Table of contents

## Table of contents

- [Web Scraping](#web-scraping)
- [Archives To Use](#archives-to-use)
    - [Archive Information](#archive-information)
- [Use New Archive](#use-new-archive)
- [Usage](#usage)
- [License](#license)

## Web Scraping

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

1. Create a scraper class in the `PaperScraper/lib/` folder.
    1. Name of the file with the scraper class should be paper_scrape_name.py` where name
       is the name of the archive with no spaces or punctuation between words, all
       letters lowercase.
    2. Class named `PaperScraperName` where Name is the name of the repository.
    3. Should output a `papers.json` file which contains a list of dicts, each dict containing
        the title, abstract, authors, category (if available), date of publication (if
        available) of a paper. The papers.json file should be saved to the
        `PaperScraper/data/archiveName` folder.
    4. Upon being initialized, the new scraper class should scrape the repository and
        compile the list of dicts for the papers. This list should then be set to the
        class variable `self.papers`.
2. From the new file import the class (PaperScraperName) into `scrape.py`
3. In scrape.py, the databases variable in `scrape_databases` function needs to be updated
    to include the new scraper. Add a dictionary entry with the key as the name of the
    repository (all lowercase, no spaces or punctuation) and the value as the name of
    the scraping class.

MAKE SURE `data/sciencedirect/paperinfo.json` IS DELETED BEFORE RUNNING FOR THE FIRST TIME WITH A NEW SEARCH QUERY

## License

[MIT](https://github.com/eth-library-lab/inDexDa/LICENSE)
