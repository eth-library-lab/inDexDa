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

## Usage

This project is divided into multiple sections to make it more modular and easier to use/modify.

### Scrape Papers

Scrape papers from an online archive in order to be able to scan them to determine whether a dataset was created or not. PaperScraper folder contains the get_papers.py script to run.

Example:
```shell
python get_papers.py --database arXiv --field ComputerScience
```
