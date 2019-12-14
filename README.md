<div align="center">
  <a href="https://www.librarylab.ethz.ch"><img src="https://www.librarylab.ethz.ch/wp-content/uploads/2018/05/logo.svg" alt="ETH Library LAB logo" height="160"></a>
  
  <br/>
  
  <p><strong>inDexDa</strong> - Natural Language Processing of academic papers for dataset identification and indexing.</p>
  
  <p>An Initiative for human-centered Innovation in the Knowledge Sphere of the <a href="https://www.librarylab.ethz.ch">ETH Library Lab</a>.</p>

</div>

## Table of contents

- [Getting Started](#getting-started)
- [Setup](#setup)
    - [Installation Instructions](#installation-instructions)
- [Usage](#usage)
    - [Configuration](#configuration)
    - [Running inDexDa](#running-indexda)
- [Contact](#contact)
- [License](#license)

## Getting Started

This project is divided into multiple sections, pipelines, to make it more modular and easier to use/modify. These are as the followings:

| Pipeline | Description |
|:-----:|:-----:|
| [PaperScraper](/PaperScraper) | Used to comb through a specified online archive of academic papers in order to find papers relating to a field, topic, or search term. Stores these papers in a MongoDB database. See PaperScraper folder for more information and usage instructions. |
| [NLP](/NLP) | From the papers found using PaperScraper, used natural language processing techniques to determine whether the papers shows a new dataset was created. If so, it stores this information within the MongoDB database for later use. See NaturalLanguageProcessing folder for more information and usage instructions. |
| Dataset Extraction | Collects information from the papers the BERT network predicts contain new datasets such as links to the dataset, type of data used, size of dataset, etc. |

## Setup

This code has been tested on a computer with following specifications:
* __OS Platform and Distribution:__ Linux Ubuntu 18.04LTS
* __CUDA/cuDNN version:__ CUDA 10.0.130, cuDNN 7.6.4
* __GPU model and memory:__ NVidia GeForce GTX 1080, 8GB
* __Python__: 3.6.8
* __TensorFlow__: 1.14

### Installation Instructions

To install the virtual environment and most of the required dependencies, run:

```bash
pip install pew
pew new inDexDa
pew in inDexDa

git clone https://github.com/ParkerEwen5441/datadex.git
cd inDexDa
./install.sh
```

Networks used in this project are run using Tensorflow backend.

## Usage

To begin running __inDexDa__ check the `args.json` file in the main directory. This contains
relevant information which will be used during the process. Please make sure to add the
following fields:

## Configuration
__inDexDa__ is configured primarily through the `args.json` file. In this file is included
a variety of options for web-scraping, network training, and dataset extraction options.
Each section is explained more thoroughly in the PaperScraper README, but the following
steps will allow you to run __inDexDa__ quickly.

1. Choose the online academic paper repository you wish to scrape in the archives_to_scrape
section. InDexDa natively supports both arXiv and ScienceDirect scraping APIs. You can
use either a single scraper or multiple scrapers in sequence.
2. Replace the default search query with your specific word or phrase. More specific search
queries will yield less results, but will run much faster.
3. If using ScienceDirect scraper, apply for an API key (https://dev.elsevier.com/apikey/manage).
Once a key has been obtained, include it in the archive_info ScienceDirect apikey field.
Also make sure to include the start and end years for the search.

## Running inDexDa
Once the `args.json` file has been configured, run the run.py file using the following flags
as desired, but only include EITHER the train or the scrape flag:

```bash
python3 run.py
    --first_time  # Must be included the first time you run inDexDa
    --scrape      # Will run inDexDa and output datasets it finds
    --train       # Will re-train the BERT network
```

## Contact

For any inquiries, use the ETH Library Lab [contact form](https://www.librarylab.ethz.ch/contact/).

## License

[MIT](LICENSE)
