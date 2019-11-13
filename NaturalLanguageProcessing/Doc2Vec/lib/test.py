from __future__ import print_function
import os
import argparse
import torch
import random
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.init as init

from dataset import Dataset
from model import NLPModel, DocClassifier


# ========================================================== #
# ===================== PARAMETERS ========================= #
# ========================================================== #
parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=8, help='input batch size')
parser.add_argument('--workers', type=int, default=12, help='number of workers')

opt = parser.parse_args()

# ===================CREATE DATASET========================= #
print('Setting up testing dataset ...')
dataset_test = Dataset(train=False)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=opt.batchSize,
                                              shuffle=True, num_workers=opt.workers,
                                              drop_last=True)

print('Testing Set: ', len(dataset_test.data))

# ========================================================== #
# ================== DOC2VEC TRAINIG ======================= #
# ========================================================== #
if not os.path.exists("../log/d2v.model"):
    print("Could not find Doc2Vec model. Make sure network was trained before testing.")
    exit()
print('Loading Doc2Vec model parameters from file ...')
doc2vec = NLPModel()

# ========================================================== #
# =============== DOC CLASSIFIER TRAINIG =================== #
# ========================================================== #
print('Testing classification network ...')

# ===================== SEED CUDA ========================== #
opt.manualSeed = random.randint(1, 10000)  # fix seed
random.seed(opt.manualSeed)

# ================== CREATE NETWORK ======================== #
network = DocClassifier()
network.cuda()

# =================== LOAD WEIGHTS ========================= #
model_path = "../log/network.pth"
try:
    network.load_state_dict(torch.load(model_path))
except:
    print("Could not find classifier model. Make sure network was trained before testing.")
    exit()

# ================== CREATE OPTIMIZER ====================== #
lrate = 0.001  # learning rate
optimizer = optim.Adam(network.parameters(), lr=lrate)

# ==================== LOSS FUNCTION ======================= #
criterion = nn.CrossEntropyLoss()

# ==================== TEST NETWORK ======================== #
network.eval()

false_positives = 0
false_negatives = 0
true_positives = 0
true_negatives = 0

with torch.no_grad():
    try:
        for i, data in enumerate(dataloader_test, 0):
            abstract, targets = data

            targets = targets.cuda()

            samples = np.empty([opt.batchSize, 50])
            for idx, doc in enumerate(abstract):
                samples[idx, :] = doc2vec.doc_vector(doc.split(' '))
            samples = torch.from_numpy(samples).cuda().float()

            classifier = network(samples)  # forward pass

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

    except RuntimeError as e:
        print(e)

total = true_positives + true_negatives + false_positives + false_negatives
accuracy = 100 * (true_positives + true_negatives) / total

if (true_positives + false_positives) != 0:
    precision = true_positives / (true_positives + false_positives)
else:
    precision = 0

if (true_positives + false_negatives) != 0:
    recall = true_positives / (true_positives + false_negatives)
else:
    recall = 0

if (true_negatives + false_positives) != 0:
    specificity = true_negatives / (true_negatives + false_positives)
else:
    specificity = 0

if (precision + recall) != 0:
    f1 = 2 * precision * recall / (precision + recall)
else:
    f1 = 0

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

with open("../log/output_stats/Doc2Vec_output_statistics.txt", "w") as f:
    f.write("=================== Doc2Vec NETWORK OUTPUT STATISTICS =====================\n\n\n")

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
