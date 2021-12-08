import numpy as np
import _pickle as pickle
from keras import Model
import tensorflow as tf
from tensorflow.keras import datasets
from src.utils.common import process_images_couple, get_datadir


def calc_embeddings(alexnet):
    # remove the last layer
    embedding_model = Model(inputs=alexnet.input, outputs=alexnet.layers[-2].output)

    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    embedding_images = np.concatenate([train_images, test_images])
    embedding_labels = np.concatenate([train_labels, test_labels])

    embedding_vds = tf.data.Dataset.from_tensor_slices((embedding_images, embedding_labels))
    embedding_vds = (embedding_vds.map(process_images_couple).batch(batch_size=32, drop_remainder=False))

    print('predicting embeddings')
    embeddings = embedding_model.predict(embedding_vds)
    print('done')

    return embeddings, embedding_labels

    # # zip together embeddings and their labels, cache in memory (maybe not necessay or maybe faster this way), shuffle, repeat forever.
    # embeddings_ds = tf.data.Dataset.zip((
    #     tf.data.Dataset.from_tensor_slices(embeddings),
    #     tf.data.Dataset.from_tensor_slices(embedding_labels)
    # ))


def save_embeddings(embeddings, labels):
    data = [embeddings, labels]

    with open(get_datadir('embeddings_labels.pkl'), 'wb') as outfile:
        pickle.dump(data, outfile, -1)


def load_embeddings():
    with open(get_datadir('embeddings_labels.pkl'), 'rb') as infile:
        result = pickle.load(infile)

    return result[0], result[1]