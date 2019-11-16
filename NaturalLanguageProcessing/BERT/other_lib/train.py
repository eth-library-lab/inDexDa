import os
import sys
import ktrain
import argparse

import tensorflow as tf

from utils import blockPrint, enablePrint
from ktrain import text
from keras.models import load_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def train_model(options):
    blockPrint()
    # load data
    datadir = '../data'
    (x_train, y_train), (x_test, y_test), preproc = text.texts_from_folder(datadir,
                                                                           maxlen=500,
                                                                           preprocess_mode='bert',
                                                                           train_test_names=['train', 'test'],
                                                                           classes=['pos', 'neg'])


    # load model
    model = text.text_classifier('bert', (x_train, y_train), preproc=preproc)

    learner = ktrain.get_learner(model,
                                 train_data=(x_train, y_train),
                                 val_data=(x_test, y_test),
                                 batch_size=options.batchSize)

    enablePrint()
    learner.fit_onecycle(2e-5, options.nepoch)

    # # save model
    save_file = '../log/bert_model.h5'
    learner.save_model(save_file)


# ========================================================== #
# ===================== PARAMETERS ========================= #
# ========================================================== #
parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=4, help='input batch size')
parser.add_argument('--nepoch', type=int, default=1, help='number of epochs')
opt = parser.parse_args()

train_model(opt)

