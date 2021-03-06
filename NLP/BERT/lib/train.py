import os
import ktrain

from ktrain import text
from termcolor import colored
from NLP.BERT.lib.utils import blockPrint, enablePrint


def train(epochs=3, batchSize=8):
    '''
    Trains the BERT model. Saves trianed BERT model in NLP/BERT/log directory.

    :params  epochs: number of epochs to train the network
             batchSize: size of batches for training
    :return  N/A
    '''
    # blockPrint()

    # ========================================================== #
    # ======================== PARAMS ========================== #
    # ========================================================== #
    ouput_msg = "Begin training the BERT network ..."
    print(colored(ouput_msg, 'cyan'))

    current_dir = os.path.dirname(os.path.abspath(__file__))
    datadir = os.path.join(current_dir, '../../../data/bert_data')
    batchSize = 4
    epochs = 1

    # ========================================================== #
    # ================= SET UP BERT NETWORK ==================== #
    # ========================================================== #
    (x_train, y_train), (x_test, y_test), preproc = text.texts_from_folder(datadir,
                                                                           maxlen=500,
                                                                           preprocess_mode='bert',
                                                                           train_test_names=['train', 'test'],
                                                                           classes=['0', '1'])

    model = text.text_classifier('bert', (x_train, y_train), preproc=preproc)

    learner = ktrain.get_learner(model,
                                 train_data=(x_train, y_train),
                                 val_data=(x_test, y_test),
                                 batch_size=batchSize)

    # ========================================================== #
    # ==================== TRAIN BERT MODEL ==================== #
    # ========================================================== #
    learner.fit_onecycle(2e-5, epochs)


    predictor = ktrain.get_predictor(learner.model, preproc=preproc)
    predictor.save('../log')
    # ========================================================== #
    # ====================== SAVE MODEL ======================== #
    # ========================================================== #
    ouput_msg = "Saving the trained BERT model in NLP/log/model.h5 ..."
    print(colored(ouput_msg, 'cyan'))

    save_dir = os.path.join(current_dir, '../log')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    save_file = os.path.join(current_dir, '../log/bert_model.h5')
    learner.save_model(save_file)

    # enablePrint()
