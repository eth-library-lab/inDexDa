# Natural Language Processing
Three different networks are tested for the pipeline which determines whether an academic
paper indicates a new dataset. The data used to train these networks were several hundred
papers which did point towards a new dataset, and 1500 which did not. The positive
examples were taken from YACVID and reference computer vision datasets, and the negative
examples were randomly sampled from arXiv over a range of subjects.

Doc2Vec and LSTM networks have been trained and the network.pth file has been provided in
their respective /log folders.

## Set Up Your Dataset
To train these networks, you can either use the dataset provided with inDexDa, or you can
create your own dataset and use that. If you wish to do the latter, you must prepare it
in the same way as the default inDexDa dataset.

__Rules for Dataset Preparation__:
1) It should be in a .json file
2) Each entry should contain the full paper abstract and the classification (0 -> doesn't
    point to dataset, 1 -> points to dataset) in that order.

Save this


## Doc2Vec
### Info
The Doc2Vec Classification network involves two seperate networks. The first assignes an
encoding to the document and the second takes the encodings of all the documents in the
dataset and trains a basic fully-connected neural network classifier.

![](https://i.stack.imgur.com/t7slV.png =300x)

In the log folder, the d2v.model is the trained Doc2Vec model while the network.pth
is the trained neural network classifier model.

### Usage
To run the


## LSTM
This network is a Recurrent Neural Network (RNN) with Long Short-Term Memory added to
better support the longer sequences of data we feed it (academic paper abstracts). The
inputs to the network are encodings of the individual words, so a Word2Vec model must
first be trained to procude this encoding.

![](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2018/08/LSTM-Autoencoder-Model.png =300x)

To see how to use this network, go to the Readme in the inDexDa/LSTM folder.

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

