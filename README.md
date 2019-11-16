# inDexDa
Natural Language Processing of academic papers for dataset identification and indexing.

## Setup

This code has been tested on a computer with following specifications:
* __OS Platform and Distribution:__ Linux Ubuntu 18.04LTS
* __CUDA/cuDNN version:__ CUDA 10.0.130, cuDNN 7.6.4
* __GPU model and memory:__ NVidia GeForce GTX 1080, 8GB
* __Python__: 3.6.8
* __PyTorch__: 1.1.0
* __TensorFlow__: 1.14

## Installation Instructions:

To install the virtual environment and most of the required dependencies, run:
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
in a MongoDB database. See PaperScraper folder for more information and usage
instructions.
2. NLP: From the papers found using PaperScraper, used natural language processing
techniques to determine whether the papers shows a new dataset was created. If so, it
stores this information within the MongoDB database for later use. See
NaturalLanguageProcessing folder for more information and usage instructions.




