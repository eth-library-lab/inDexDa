# inDexDa
Natural Language Processing of academic papers for dataset identification and indexing.

## Setup

This code has been tested on a computer with following specifications:
* __OS Platform and Distribution:__ Linux Ubuntu 16.04LTS
* __CUDA/cuDNN version:__ CUDA 10.1.168, cuDNN 7.1.4
* __GPU model and memory:__ NVidia GeForce GTX 740M, 2GB
* __Python__: 3.5.2
* __PyTorch__: 1.1.0

## Installation Instructions:

To install the virtual environment and all required dependencies, run:
```bash
./install.sh
```

## Setup
This project uses several config files in order to run.

* PaperScraper/config/repository/config.json
* PaperScraper/data/repository/field/links.txt

The first of these contains information specific to scraping arXiv such as what topic, the starting year and month for the papers you want to scrape, and the ending
year and month

## Usage

This project is divided into multiple sections to make it more modular and easier to use/modify. The first section involves gathering information about publications
from online repositories and storing it in a MongoDB database. The second portion trains a neural network to read paper abstracts and determine whether or not a
new dataset was created. If it was, the paper entry within the MongoDB database is updated with an entry indicating as such.

### Scrape Papers

PaperScraper folder contains the run.py script. The script will confirm which repository you wish to use followed by a series of questions. The first is a confirmation
as to whether the user wishes to scrape the given repository (yes by default) followed by whether the user wants to update the database. The database uses the links to
the webpage containing information on the paper (located in /PaperScraper/data/) as the means of updating. The following example will use the arXiv online repository and find the links to all papers within the Computer Science field.

Example:
```shell
python run.py --database arXiv --field ComputerScience
```

