from __future__ import print_function
import argparse
import torch
import random
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.init as init

from dataset import Dataset
from model import NLPModel, DocClassifier


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


# ========================================================== #
# ===================== PARAMETERS ========================= #
# ========================================================== #
parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=8, help='input batch size')
parser.add_argument('--workers', type=int, default=12, help='number of workers')
parser.add_argument('--nepoch', type=int, default=5, help='number of epochs')
parser.add_argument('--model', type=str, default='', help='optional reload model path')
parser.add_argument('--no-nlp-train', dest='nlp', action='store_false')
parser.set_defaults(nlp=True)

opt = parser.parse_args()

# ===================CREATE DATASET========================= #
print('Setting up datasets ...')
dataset = Dataset(train=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True,
                                         num_workers=8, drop_last=True)
dataset_test = Dataset(train=False)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=8, shuffle=True,
                                              num_workers=8, drop_last=True)

print('Training Set: ', len(dataset.data))
print('Testing Set: ', len(dataset_test.data))
len_dataset = len(dataset)
len_dataset_test = len(dataset_test)

# ========================================================== #
# ================== DOC2VEC TRAINIG ======================= #
# ========================================================== #
if opt.nlp is True:
    print('Training Doc2Vec network for document encodings ...')
    corpus = [item[0] for item in dataset.all_samples]
    doc2vec = NLPModel(epochs=100, method='PVDM')
    doc2vec.train(corpus)
else:
    print('Loading Doc2Vec model parameters from file ...')
    doc2vec = NLPModel()

# ========================================================== #
# =============== DOC CLASSIFIER TRAINIG =================== #
# ========================================================== #
print('Training classification network ...')

# ===================== SEED CUDA ========================== #
opt.manualSeed = random.randint(1, 10000)  # fix seed
random.seed(opt.manualSeed)

# ================== CREATE NETWORK ======================== #
network = DocClassifier()
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
            abstract, target = data

            target = target.cuda()

            # Transform document text into vectorized encoding
            samples = np.empty([opt.batchSize, 50])
            for idx, doc in enumerate(abstract):
                samples[idx, :] = doc2vec.doc_vector(doc.split(' '))
            samples = torch.from_numpy(samples).cuda().float()

            classifier = network(samples)  # forward pass

            loss = criterion(classifier, target)
            loss.backward()
            optimizer.step()  # gradient update

            total_batches = len_dataset / opt.batchSize - 1
            print('[%d: %d/%d] train loss:  %f ' % (epoch, i, total_batches, loss.item()))

    except RuntimeError as e:
        print(e)

    # save last network
    save_path = '/home/parker/code/datadex/inDexDa/Doc2Vec/log/network.pth'
    torch.save(network.state_dict(), save_path)

# ==================== TEST NETWORK ======================== #
    network.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        try:
            for i, data in enumerate(dataloader_test, 0):
                abstract, target = data

                target = target.cuda()

                samples = np.empty([opt.batchSize, 50])
                for idx, doc in enumerate(abstract):
                    samples[idx, :] = doc2vec.doc_vector(doc.split(' '))
                samples = torch.from_numpy(samples).cuda().float()

                classifier = network(samples)  # forward pass
                loss = criterion(classifier, target)

                # Accuracy of the network
                _, predicted = torch.max(classifier.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

                total_batches = len_dataset_test / opt.batchSize - 1
                print('[%d: %d/%d] val loss:  %f ' % (epoch, i, total_batches,
                                                      loss.item()))

        except RuntimeError as e:
            print(e)

    accuracy = 100 * correct / total
    print("Epoch: {}/{} | Accuracy: {}".format(epoch, opt.nepoch, accuracy))
