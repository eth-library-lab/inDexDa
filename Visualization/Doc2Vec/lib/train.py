import os
import torch
import numpy as np

from Doc2Vec.lib.dataset import Dataset
from Doc2Vec.lib.model import NLPModel


def train():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(current_dir, '../../dataset_names.json')

    # ===================CREATE DATASET========================= #
    print('Setting up datasets ...')
    dataset = Dataset(train=True, datapath=dataset_dir)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True,
                                             num_workers=8, drop_last=False)
    len_dataset = len(dataset)

    # ================== DOC2VEC TRAINING ====================== #
    print('Training Doc2Vec network for document encodings ...')

    vec_size = 5
    corpus = [item[1] for item in dataset.data]
    doc2vec = NLPModel(epochs=100, method='PVDM', vec_size=vec_size)
    doc2vec.train(corpus)

    # =================== GET DOC VECTORS ====================== #
    doc_names = []
    doc_encodings = np.empty([len(dataset), vec_size])
    for i, data in enumerate(dataloader, 0):
        name, keywords = data

        actual_keywords = []
        for tupl in keywords:
            actual_keywords.append(tupl[0])
        keywords = actual_keywords

        # Transform document text into vectorized encoding
        doc_names.append(name[0])
        doc_encodings[i, :] = doc2vec.doc_vector(keywords)

    return doc_names, doc_encodings


if __name__ == '__main__':
    docs = train()
    input(docs[0])
