import os
import json
import math
import ktrain
import NLP.utils.command_line as progress

from ktrain import text
from termcolor import colored
from utils import removeDuplicates


def test(datadir, batchSize=6):
    '''
    Predicts whether or not an abstract indicates a new dataset.

    :param datadir: directory of evaulation examples
    :return classification: list of ints
    '''
    # ========================================================== #
    # ======================== PARAMS ========================== #
    # ========================================================== #
    current_dir = os.path.dirname(os.path.abspath(__file__))
    traindir = os.path.join(current_dir, '../../../data/bert_data')

    # ========================================================== #
    # ================= GET EVALUATION DATA ==================== #
    # ========================================================== #
    output_msg = 'Setting up BERT network for classification ...'
    print(colored(output_msg, 'cyan'))

    if not os.path.exists(traindir):
        error = ('Data in directory inDexDa/data/bert_data has either been',
                 ' deleted or is formatted incorrectly. Refer to original',
                 ' data supplied in the repo for proper formatting.')
        print(colored(error, 'red'))
        raise Exception(error)

    if not os.path.exists(datadir):
        error = ('Data directory for evaluation data does not exist. Make sure',
                 ' that directory and eval.json file exist at: {}'.format(datadir))
        print(colored(error, 'red'))
        raise Exception(error)

    with open(datadir, 'r') as f:
        contents = f.read()
        raw = json.loads(contents)
        eval_papers = [paper["Abstract"] for paper in raw]

    # ========================================================== #
    # ================= SET UP BERT NETWORK ==================== #
    # ========================================================== #
    (x_train, y_train), (x_test, y_test), preproc = text.texts_from_folder(traindir,
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
    # =============== LOAD PRETRAINED BERT MODEL =============== #
    # ========================================================== #
    output_msg = 'Loading the pretrained BERT network ...'
    print(colored(output_msg, 'cyan'))

    load_file = os.path.join(current_dir, '../log/bert_model.h5')
    try:
        learner.load_model(load_file)
    except:
        error = 'Something went wrong when trying to load the weights for the BERT model.'
        print(colored(error, 'red'))
        exit()

    predictor = ktrain.get_predictor(learner.model, preproc)

    # ========================================================== #
    # ======================== PREDICT ========================= #
    # ========================================================== #
    output_msg = 'Predicting if new datasets are presented ...'
    print(colored(output_msg, 'cyan'))

    prediction = predictor.predict(eval_papers)

    results = []
    for idx, paper in enumerate(eval_papers):
        if prediction[idx] == '0':
            results.append({"Abstract": paper, "Prediction": "No Dataset"})
        elif prediction[idx] == '1':
            results.append({"Abstract": paper, "Prediction": "Dataset Detected"})

    # ========================================================== #
    # ================== INFO ABOUT DATASETS =================== #
    # ========================================================== #
    output_msg = "Finalizing BERT Outputs ..."
    print(colored(output_msg, 'cyan'))

    dataset_papers = []
    for idx, result in enumerate(results):
        progress.printProgressBar(idx + 1, math.ceil(len(results)),
                                  prefix='Progress :', suffix='Complete',
                                  length=30)
        for paper in raw:
            if result["Abstract"] == paper["Abstract"] and "Dataset Detected" in result["Prediction"]:
                paper.update({"Prediction": result["Prediction"]})
                dataset_papers.append(paper)

    # ========================================================== #
    # ========================= SAVE =========================== #
    # ========================================================== #
    # Remove duplicate entries
    dataset_papers = removeDuplicates(dataset_papers)

    output_msg = 'Saving results ...'
    print(colored(output_msg, 'cyan'))
    outputdir = os.path.join(current_dir, '../../../data/results.json')
    with open(outputdir, 'w') as f:
        json.dump(dataset_papers, f, indent=4)


if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    datadir = os.path.join(current_dir, '../../../data/results.json')
    test(datadir)
