from __future__ import print_function
import os
import sys
import argparse
import torch
import random
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init

LSTM_PACKAGE = '..'
LSTM_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(),
                                                         os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(LSTM_DIR, LSTM_PACKAGE)))

from utils.util import findMax, roundToNextFifty
from dataset import Dataset
from model import Word2VecModel, LSTMClassifier


def weights_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)


# ========================================================== #
# ===================== PARAMETERS ========================= #
# ========================================================== #
parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--workers', type=int, default=12, help='number of workers')
parser.add_argument('--nepoch', type=int, default=20, help='number of epochs')
parser.add_argument('--model', type=str, default='', help='optional reload model path')
parser.add_argument('--word2vec-train', dest='nlp', action='store_false')
parser.set_defaults(nlp=True)

opt = parser.parse_args()

# ========================================================== #
# =================== INIT DATASETS ======================== #
# ========================================================== #
print('Setting up datasets ...')
dataset = Dataset(train=True)
dataset_test = Dataset(train=False)

print('Training Set: ', len(dataset.data))
print('Testing Set: ', len(dataset_test.data))
len_dataset = len(dataset)
len_dataset_test = len(dataset_test)

# ========================================================== #
# ================== WORD2VEC TRAINIG ====================== #
# ========================================================== #
if opt.nlp is True:
    print('Training Word2Vec network for document encodings ...')
    corpus = [item[0].split(' ') for item in dataset.all_samples]
    word2vec = Word2VecModel()
    word2vec.train(corpus)
else:
    print('Loading Word2Vec model parameters from file ...')
    word2vec = Word2VecModel()

# ========================================================== #
# ================= DATASET STATISTICS ===================== #
# ========================================================== #
lengths = []
for sample in dataset.all_samples:
    lengths.append(len(sample[0].split(' ')))
max_seq_length = int(roundToNextFifty(findMax(lengths)))
print('Input sequence length for LSTM is {}'.format(max_seq_length))

# ========================================================== #
# ============== VECTORIZED DATASET SETUP ================== #
# ========================================================== #
print('Generating vectorized dataset ...')
dataset.LSTMSequenceGen(max_seq_length, 5, word2vec)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True,
                                         num_workers=8, drop_last=True)

dataset_test.LSTMSequenceGen(max_seq_length, 5, word2vec, train=False)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=opt.batchSize,
                                              shuffle=True, num_workers=8, drop_last=True)

# ========================================================== #
# =============== DOC CLASSIFIER TRAINING ================== #
# ========================================================== #
print('Training LSTM Classification network ...')

# ===================== SEED CUDA ========================== #
opt.manualSeed = random.randint(1, 10000)  # fix seed
random.seed(opt.manualSeed)

# ================== CREATE NETWORK ======================== #
network = LSTMClassifier(5, 5, opt.batchSize, 2, max_seq_length)
network.cuda()
network.apply(weights_init)  # initialization of the weight

# =================== LOAD WEIGHTS ========================= #
if opt.model != '':
    network.load_state_dict(torch.load(opt.model))

# ================== CREATE OPTIMIZER ====================== #
lrate = 0.001  # learning rate
optimizer = optim.Adam(network.parameters(), lr=lrate)

# ==================== LOSS FUNCTION ======================= #
criterion = nn.CrossEntropyLoss()

# =================== START TRAINING ======================= #
for epoch in range(opt.nepoch):
    network.train()

    # learning rate schedule
    if epoch == opt.nepoch / 2:
        optimizer = optim.Adam(network.parameters(), lr=lrate / 10.0)

    try:
        for i, data in enumerate(dataloader, 0):
            optimizer.zero_grad()
            abstracts, targets = data

            targets = targets.cuda()
            abstracts = abstracts.cuda().float()

            classifier = network(abstracts)  # forward pass

            loss = criterion(classifier, targets)
            loss.backward()
            optimizer.step()  # gradient update

            total_batches = len_dataset / opt.batchSize - 1
            # print('[%d: %d/%d] train loss:  %f ' % (epoch, i, total_batches, loss.item()))

    except RuntimeError as e:
        print(e)

    # save last network
    save_file = '/home/parker/code/datadex/inDexDa/LSTM/log/network.pth'
    torch.save(network.state_dict(), save_file)

# ==================== TEST NETWORK ======================== #
    network.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        try:
            for i, data in enumerate(dataloader_test, 0):
                abstracts, targets = data

                targets = targets.cuda()
                abstracts = abstracts.cuda().float()

                classifier = network(abstracts)  # forward pass

                loss = criterion(classifier, targets)

                # Accuracy of the network
                _, predicted = torch.max(classifier.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

                total_batches = len_dataset_test / opt.batchSize - 1
                # print('[%d: %d/%d] val loss:  %f ' % (epoch, i, total_batches,
                #                                       loss.item()))

        except RuntimeError as e:
            print(e)

    accuracy = 100 * correct / total
    print("Epoch: {}/{} | Accuracy: {}".format(epoch, opt.nepoch, accuracy))
