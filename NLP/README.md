# Natural Language Processing
Three different networks were tested for the pipeline which determines whether an academic
paper indicates a new dataset. The data used to train these networks were several hundred
papers which did point towards a new dataset, and 1500 which did not. The positive
examples were taken from YACVID and reference computer vision datasets, and the negative
examples were randomly sampled from arXiv over a range of subjects. The best performing
network, BERT, was chosen as the network of choice for this project.


## Set Up Your Dataset
To train these networks, you can either use the dataset provided with inDexDa, or you can
create your own dataset and use that. If you wish to do the latter, you must prepare it
in the same way as the default inDexDa dataset.

__Rules for Dataset Preparation__:
1) It should be in a .json file
2) Each entry should contain the full paper abstract and the classification (0 -> doesn't
    point to dataset, 1 -> points to dataset) in that order.


## BERT
BERT is the Bidirectional Encoder Representations from Transforms language model. It was
recently published by researchers in Google's AI Learning division. The complexity of
BERT cannot be explained here, but Google allows you to download the pre-trained network
and then fine-tuning can be done for the user-specific task. We train it on the same
data as the other two networks, however the input structure is slightly different.

![](https://miro.medium.com/max/876/0*ViwaI3Vvbnd-CJSQ.png=300x)

### Usage
BERT takes as inputs .tsv files which are set up in a particular way. Make sure that
the dataset you want to use is set up in the same way as the default inDexDa dataset (
place it in the inDexDa/data folder and name it dataset.json). Next, run the program
runBERT.py. This script will first set up the .tsv files and then start training
BERT with those files.

