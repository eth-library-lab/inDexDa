import os
import sys
import ktrain
import argparse

import tensorflow as tf

from utils import blockPrint, enablePrint
from ktrain import text
from keras.models import load_model


def test(sentences):
    '''
    Predicts whether or not an abstract indicates a new dataset.

    :param sentences: list of strings
    :return classification: list of ints
    '''
    blockPrint()
    # load data
    datadir = '../data'
    (x_train, y_train), (x_test, y_test), preproc = text.texts_from_folder(datadir,
                                                                           maxlen=500,
                                                                           preprocess_mode='bert',
                                                                           train_test_names=['train', 'test'],
                                                                           classes=[1, 0])

    model = text.text_classifier('bert', (x_train, y_train), preproc=preproc)

    learner = ktrain.get_learner(model,
                                 train_data=(x_train, y_train),
                                 val_data=(x_test, y_test),
                                 batch_size=6)

    load_file = '../log/bert_model.h5'
    learner.load_model(load_file)

    # predict output
    predictor = ktrain.get_predictor(learner.model, preproc)
    data = ['This abstract shows we have published a new dataset that is now available online.'
            ' It contains 150 Gb of images from traffic scenes.',
            'I am here today to talk about my research where we used data to determine that the'
            ' sun does not really exist. We know this because we used Johns dataset to confirm'
            'our bias. You are all dumb.',
            'We love pineapple. No idea why, we just do.']

    prediction = predictor.predict(data)

    enablePrint()
    print(prediction)
