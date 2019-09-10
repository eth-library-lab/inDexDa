# inDexDa
Natural Language Processing of academic papers for dataset identification and indexing.

## Setup

This code has been tested on a computer with following specifications:
* __OS Platform and Distribution:__ Linux Ubuntu 18.04LTS
* __CUDA/cuDNN version:__ CUDA 10.1.168
* __GPU model and memory:__ NVidia GeForce GTX 1080, 8GB
* __Python__: 3.6.8
* __PyTorch__: 1.1.0

## Installation Instructions:

To install the virtual environment and all required dependencies, run:
```bash
./install.sh
```
Among other things, this will install a MongoDB database. This is what is used to store
information relating to this project. It can be deleted at any time following the
instructions: https://docs.mongodb.com/v3.2/tutorial/install-mongodb-on-ubuntu/

## Usage

This project is divided into multiple sections to make it more modular and easier to
use/modify. TThey are as follows:

1. PaperScaper: Used to comb through a specified online archive of academic papers
in order to find papers relating to a field, topic, or search term. Stores these papers
in a MongoDB database.
2. NLP: From the papers found using PaperScraper, used natural language processing
techniques to determine whether the papers shows a new dataset was created. If so, it
stores this information within the MongoDB database for later use.

### Scrape Papers

__SETUP__

This portion of the project uses a config file in order to run.

* PaperScraper/config/repository/config.json

It contains information specific to scraping the selected database such as API key
(for ScienceDirect and IEEE Xplore), query keyword, and search dates to restrict the search
results.

__USE SUPPORTED DATABASES__

Three databases are supported in the vanilla version of inDexDa. They are: arXiv,
ScienceDirect, and IEEE Xplore. To use these databases, follow the instructions below.

The PaperScraper folder contains the run.py script which allows the user to both search
through the online databases for a specfic query (default is 'dataset') as well as update
a database (storage of all scraped papers). Before running the script, verify the
parameters within the _database/config.json_ file. Each online database requires unique
paramaters for their specific web scraping API.

ScienceDirect and IEEEXplore both require API keys, so the user must apply to these at the
following web sites:

* ScienceDirect: https://dev.elsevier.com/apikey/manage
* IEEE Xplore: https://developer.ieee.org/

Once API keys have been registered and the appropriate _config.json_ files updates, run
the following script.

Example:
```shell
python run.py --database database
```
It will first confirm with the user the online repository they wish to scrape. By
confirming, the web scraper will use the database's appropriate API to collect papers
relating to the query term within the _config.json_.

Follwoing this, the script will ask the user to update the Mongo database. The script
will also currate the Mongo database by removing any papers which are duplicates.

__USE NEW DATABASE__

inDexDa also allows users to use online or local repositories which are not natively
supported. To do this, the user must first create the PaperScraper class for the
site in the _PaperScraper/lib/_ folder (following the naming convention). Additionally,
a file must be created for the parameters (_PaperScraper/config/database/config.json_)
and the name/class should be added to the dictionary in the
_PaperScraper/scrape_database.py_ file.

The created class should take as input the config file for initialization, and the
database.papers variable should call a scrape4papers which returns a list of dictionaries,
each dictionary containing the Title, Abstract, Authors, Category, and Date of publication
of a paper. This output is then saved to _PaperScraper/data/database/papers.json_.

This will allow the following portion of inDexDa to use papers from the new repository.

### Natural Language Processing
