import numpy as np
import _pickle as pickle
from keras import Model
from src.data.cifar10 import cifar10_complete_resized
from src.utils.common import get_datadir


def calc_embeddings(alexnet):
    # remove the last layer
    embedding_model = Model(inputs=alexnet.input, outputs=alexnet.layers[-2].output)

    embedding_vds = cifar10_complete_resized().batch(batch_size=32, drop_remainder=False)

    print('predicting embeddings')
    embeddings = embedding_model.predict(embedding_vds)
    embedding_labels = np.concatenate([y for x, y in embedding_vds], axis=0)
    return embeddings, embedding_labels


def save_embeddings(embeddings, labels):
    data = [embeddings, labels]

    with open(get_datadir('embeddings_labels.pkl'), 'wb') as outfile:
        pickle.dump(data, outfile, -1)


def load_embeddings():
    with open(get_datadir('embeddings_labels.pkl'), 'rb') as infile:
        result = pickle.load(infile)

    return result[0], result[1]
