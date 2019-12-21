<div align="center">
  <a href="https://www.librarylab.ethz.ch"><img src="https://www.librarylab.ethz.ch/wp-content/uploads/2018/05/logo.svg" alt="ETH Library LAB logo" height="160"></a>

  <br/>

  <p><strong>inDexDa</strong> - Natural Language Processing of academic papers for dataset identification and indexing.</p>

  <p>An Initiative for human-centered Innovation in the Knowledge Sphere of the <a href="https://www.librarylab.ethz.ch">ETH Library Lab</a>.</p>

</div>

## Table of contents

- [Dataset Information Extraction](#datset-information-extraction)
- [Setup with Supported Archives](#setup-with-supported-archives)
    - [Archive Information](#archive-information)
    - [Scraping](#scraping)
    - [Information Extracted](#information-extracted)
- [Use New Archive](#use-new-archive)
- [License](#license)


# Dataset Information Extraction
## Setup with Supported Archives
For this pipeline, we use only the papers predicted to point to datasets output
by the BERT network. Like with PaperScraper, we must scrape the archive where
the paper was originally from, this time to acquire the full body text rather
than just snippits of information.

### Archive Information
Like PaperScraper, the args.json is where the supported scraping archives can
be found. Additionally, the indicated archive to scrape for PaperScraper is
also used here (for logical consistency within the entire pipeline).

The user can specify either one or multiple online repositories to scrape from by modifying
the number of entries in the archive tag. The syntax is as follows:

```shell
{"id": "0x", "archive": "name"}
    # x should be an integer between 1-9
    # name is the name of the online paper repository, all lowercase, no spaces
```

### Scraping
In this section scraping must be done differently. Instead of collecting meta
information on papers, we must scrpae their body text and compile it into a
single string. The two natively supported archives, ArXiv and ScienceDirect,
have different scraping pipelines for this as explained below:

* AArXiv: The API does not include body text information, so we must download
            the paper's pdf, convert it to text, and then clean up this text
            for later data extraction. Logic for this scraper can be found in
            DatasetIndexing/lib/arxiv_scraper.py
* ScienceDirect: The Article Retrieval API works to get the entire paper,
            however, due to issues with parsing, only the xml version contains
            all the text, the json version has only the abstract. Once the xml
            is acquired, it is parsed and the body text extracted. To use this
            API, an API Key is still required

* ScienceDirect API Key Application: https://dev.elsevier.com/apikey/manage

### Information Extracted
Currently, the only information extracted from the papers are possible dataset
urls and possible dataset names. The logic for each section can be seen in the
scraper functions.

As an open-source project, we encourage contributers to add additional features,
especially at this portion of the pipeline to help offer users more information
on possible datasets.


## Use New Archive
inDexDa also allows users to use online or local repositories which are not natively
supported. To do this, the following must be done:

1. Create a scraper class in the DatasetIndexing/lib/_ folder.
    1. Name of the file with the scraper class should be name_scraper.py where name
       is the name of the archive with no spaces or punctuation between words, all
       letters lowercase.
    2. Class named NameScraper where Name is the name of the repository.
    3. Each class should be fed one paper at a time (paper read from data/results.json)
        and should return an updated dict entry of the paper, with updates consisting
        of information extracted from the paper.
    4. Class function _extract_ should run the entire scraping pipeline and
        return the updated paper entry (see arXiv or ScienceDirect scrapers for
        an example)
2. From the new file import the class (PaperScraperName) into infoExtraction.py
3. In infoExtraction.py, the databases variable in ExtractInfoFromPaper().extract
     needs to be updated to include the new scraper. Add a dictionary entry with
     the key as the name of the repository (all lowercase, no spaces or
     punctuation) and the value as the name of the scraping class.


## License

[MIT](https://github.com/eth-library-lab/inDexDa/LICENSE)
