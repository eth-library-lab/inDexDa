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
# ================== WORD2VEC TRAINING ===================== #
# ========================================================== #
try:
    print('Loading Word2Vec model parameters from file ...')
    word2vec = Word2VecModel()
except:
    print("Word2Vec model not found. Make sure to train model before testing.")
    exit()

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
print('Generating vectorized testing dataset ...')
dataset_test.LSTMSequenceGen(max_seq_length, 5, word2vec, train=False)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=opt.batchSize,
                                              shuffle=True, num_workers=opt.workers,
                                              drop_last=True)

# ========================================================== #
# ==============-= DOC CLASSIFIER TESTING ================== #
# ========================================================== #
print('Testing LSTM Classification network ...')

# ===================== SEED CUDA ========================== #
opt.manualSeed = random.randint(1, 10000)  # fix seed
random.seed(opt.manualSeed)

# ================== CREATE NETWORK ======================== #
network = LSTMClassifier(5, 5, opt.batchSize, 2, max_seq_length)
network.cuda()
network.apply(weights_init)  # initialization of the weight

# =================== LOAD WEIGHTS ========================= #
model_path = "../log/network.pth"
try:
    network.load_state_dict(torch.load(model_path))
except:
    print("LSTM model could not be found. Make sure to train model before testing.")
    exit()

# ==================== TEST NETWORK ======================== #
network.eval()

false_positives = 0
false_negatives = 0
true_positives = 0
true_negatives = 0

with torch.no_grad():
    try:
        for i, data in enumerate(dataloader_test, 0):
            abstracts, targets = data

            targets = targets.cuda()
            abstracts = abstracts.cuda().float()

            classifier = network(abstracts)  # forward pass

            # Accuracy of the network
            _, predicted = torch.max(classifier.data, 1)

            for idx, _ in enumerate(predicted):
                if predicted[idx] == 1 and targets[idx] == 1:
                    true_positives += 1
                if predicted[idx] == 1 and targets[idx] == 0:
                    false_positives += 1
                if predicted[idx] == 0 and targets[idx] == 1:
                    false_negatives += 1
                if predicted[idx] == 0 and targets[idx] == 0:
                    true_negatives += 1

            total_batches = len_dataset_test / opt.batchSize - 1

    except RuntimeError as e:
        print(e)

total = true_positives + true_negatives + false_positives + false_negatives
accuracy = 100 * (true_positives + true_negatives) / total
precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)
specificity = true_negatives / (true_negatives + false_positives)
f1 = 2 * precision * recall / (precision + recall)

print("\nTESTING STATISTICS ")
print("Accuracy    : {}".format(accuracy))
print("Precision   : {}".format(precision))
print("Recall      : {}".format(recall))
print("Specificity : {}".format(specificity))
print("F1 Score    : {}".format(f1))


# ========================================================== #
# ================ PRINT OUTPUT STATS ====================== #
# ========================================================== #
if not os.path.exists("../log/output_stats"):
    os.mkdir("../log/output_stats")

with open("../log/output_stats/LSTM_output_statistics.txt", "w") as f:
    f.write("==================== LSTM NETWORK OUTPUT STATISTICS =======================\n\n\n")

    f.write("                                                  ACTUAL                       \n")
    f.write("                                  Postives                       Negatives     \n")
    f.write("                      =========================================================\n")
    f.write("   P                  |                              |                        |\n")
    f.write("   R     Positives    |        TRUE POSITIVES        |      FALSE POSITIVES   |\n")
    f.write("   E                  |                              |                        |\n")
    f.write("   D                  =========================================================\n")
    f.write("   I                  |                              |                        |\n")
    f.write("   C     Negatives    |        FALSE NEGATIVES       |       TRUE NEGATIVES   |\n")
    f.write("   T                  |                              |                        |\n")
    f.write("                      =========================================================\n\n\n")

    f.write(" True Positives  : {}\n".format(true_positives))
    f.write(" True Negatives  : {}\n".format(true_negatives))
    f.write(" False Positives : {}\n".format(false_positives))
    f.write(" False Negatives : {}\n\n".format(false_negatives))

    f.write(" Accuracy    : {}\n".format(accuracy))
    f.write(" Precision   : {}\n".format(precision))
    f.write(" Recall      : {}\n".format(recall))
    f.write(" Specificity : {}\n".format(specificity))
    f.write(" F1 Score    : {}\n".format(f1))
