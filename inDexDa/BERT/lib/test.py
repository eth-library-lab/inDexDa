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
parser.add_argument('--model', type=str, default='', help='optional reload model path')

opt = parser.parse_args()

# ===================CREATE DATASET========================= #
print('Setting up testing dataset ...')
dataset_test = Dataset(train=False)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=opt.batchSize,
                                              shuffle=True, num_workers=opt.workers,
                                              drop_last=True)

print('Testing Set: ', len(dataset_test.data))
len_dataset_test = len(dataset_test)

# ========================================================== #
# ================== DOC2VEC TRAINIG ======================= #
# ========================================================== #
try:
    print('Loading Doc2Vec model parameters from file ...')
    doc2vec = NLPModel()
except:
    print('Could not find Doc2Vec model. Make sure the network has been trained.')
    exit()

# ========================================================== #
# =============== DOC CLASSIFIER TRAINING ================== #
# ========================================================== #
print('Testing classification network ...')

# ===================== SEED CUDA ========================== #
opt.manualSeed = random.randint(1, 10000)  # fix seed
random.seed(opt.manualSeed)

# ================== CREATE NETWORK ======================== #
network = DocClassifier()
network.cuda()

# ========================================================== #
# =================== LOAD WEIGHTS ========================= #
# ========================================================== #
if opt.model != '':
    network.load_state_dict(torch.load(opt.model))
else:
    network.load_state_dict(torch.load('../log/network.pth'))

# ========================================================== #
# ==================== TEST NETWORK ======================== #
# ========================================================== #
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

            # Accuracy of the network
            _, predicted = torch.max(classifier.data, 1)
            total += target.size(0)
            # correct += (predicted == target).sum().item()
            correct += torch.sum((predicted == target))

    except RuntimeError as e:
        print(e)

accuracy = 100 * correct / total

# ========================================================== #
# =============== OUTPUT RESULT STATISTICS ================= #
# ========================================================== #
true_positives = 0
true_negatives = 0
false_positives = 0
false_negatives = 0

true_positives_text = []
true_negatives_text = []
false_positives_text = []
false_negatives_text = []

with torch.no_grad():
    try:
        for i, data in enumerate(dataloader_test, 0):
            abstracts, targets = data

            targets = targets.cuda()

            samples = np.empty([opt.batchSize, 50])
            for idx, doc in enumerate(abstract):
                samples[idx, :] = doc2vec.doc_vector(doc.split(' '))
            samples = torch.from_numpy(samples).cuda().float()

            classifier = network(samples)  # forward pass

            _, predicted = torch.max(classifier.data, 1)

            for idx, item in enumerate(predicted, 0):
                if predicted[idx] == 0 and targets[idx] == 0:
                    true_negatives += 1
                    true_negatives_text.append(abstracts[idx])

                elif predicted[idx] == 0 and targets[idx] == 1:
                    false_negatives += 1
                    false_negatives_text.append(abstracts[idx])

                elif predicted[idx] == 1 and targets[idx] == 0:
                    false_positives += 1
                    false_positives_text.append(abstracts[idx])

                elif predicted[idx] == 1 and targets[idx] == 1:
                    true_positives += 1
                    true_positives_text.append(abstracts[idx])

    except RuntimeError as e:
        print(e)

    if (true_positives + true_negatives) != 0:
        precision = true_positives / (true_positives + true_negatives)
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
        f1 = (2 * precision * recall) / (precision + recall)
    else:
        f1 = 0

    print("Total Accuracy: {}".format(accuracy))
    print('Precision: {}'.format(precision))
    print('Recall: {}'.format(recall))
    print('Specificity: {}'.format(specificity))
    print('F1 Score: {}'.format(f1))

print('Creating files for network output statistics...')

if not os.path.isdir("../log/output_stats"):
    os.mkdir('../log/output_stats')

with open('../log/output_stats/Doc2Vec_output_statistics.txt', 'w') as f:
    f.write('-- OUTPUT STATISTICS FROM DOC2VEC NETWORK -- \n\n')
    f.write('True Positives:  {}\n'.format(true_positives))
    f.write('False Positives:  {}\n'.format(false_positives))
    f.write('True Negatives:  {}\n'.format(true_negatives))
    f.write('False Negatives:  {}\n\n\n'.format(false_negatives))

    f.write('                                      ACTUAL                     \n')
    f.write('                       POSITIVES                 NEGATIVES       \n')
    f.write('               ==================================================\n')
    f.write('P              |                         |                      |\n')
    f.write('R   POSITIVES  |     True Positives      |     False Positives  |\n')
    f.write('E              |                         |                      |\n')
    f.write('D              |=========================|======================|\n')
    f.write('I              |                         |                      |\n')
    f.write('C   NEGATIVES  |     False Negatives     |     True Negatives   |\n')
    f.write('T              |                         |                      |\n')
    f.write('               ==========================|=======================\n\n')

    f.write('Precision is how many were classified as true when they were\n')
    f.write('   actually true.\n')
    f.write('Precision: {}\n\n'.format(precision))

    f.write('Recall tells us what proportion of examples that were positive\n')
    f.write('   were classified as such by the algorithm.\n')
    f.write('Recall: {}\n\n'.format(recall))

    f.write('Specificity is how many were classified as false when they were\n')
    f.write('   actually false.\n')
    f.write('Specificity:  {}\n\n'.format(specificity))

    f.write('The F1 score is the harmonic mean between specificity and recall.\n')
    f.write('   If either recall or specificity is really small, the F1 score.\n')
    f.write('   will reflect this by also being small.\n')
    f.write('F1 Score:  {}\n\n'.format(f1))

with open('../log/output_stats/true_positives_abstracts.txt', 'w') as f:
    f.write('-- ABSTRACTS FOR TRUE POSITIVES FROM DOC2VEC NETWORK -- \n\n')
    for item in true_positives_text:
        f.write(item + '\n\n')

with open('../log/output_stats/true_negatives_abstracts.txt', 'w') as f:
    f.write('-- ABSTRACTS FOR TRUE NEGATIVES FROM DOC2VEC NETWORK -- \n\n')
    for item in true_negatives_text:
        f.write(item + '\n\n')

with open('../log/output_stats/false_positives_abstracts.txt', 'w') as f:
    f.write('-- ABSTRACTS FOR FALSE POSITIVES FROM DOC2VEC NETWORK -- \n\n')
    for item in false_positives_text:
        f.write(item + '\n\n')

with open('../log/output_stats/false_negatives_abstracts.txt', 'w') as f:
    f.write('-- ABSTRACTS FOR FALSE NEGATIVES FROM DOC2VEC NETWORK -- \n\n')
    for item in false_negatives_text:
        f.write(item + '\n\n')
