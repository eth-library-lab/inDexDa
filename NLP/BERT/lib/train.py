import os
import ktrain

from ktrain import text
from NLP.BERT.lib.utils import blockPrint, enablePrint


def train():
    '''
    Trains the BERT model.

    :param datadir: directory of evaulation examples
    :return classification: list of ints
    '''
    blockPrint()

    # ========================================================== #
    # ======================== PARAMS ========================== #
    # ========================================================== #
    current_dir = os.path.dirname(os.path.abspath(__file__))
    datadir = os.path.join(current_dir, '../../../data/bert_data')
    batchSize = 8
    epochs = 3

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

    # ========================================================== #
    # ====================== SAVE MODEL ======================== #
    # ========================================================== #
    save_file = os.path.join(current_dir, '../log/bert_model.h5')
    learner.save_model(save_file)

    enablePrint()
