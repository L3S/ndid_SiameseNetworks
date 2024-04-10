import csv
import time
import bz2
import pickle
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from tensorboard.plugins import projector
from google.protobuf import text_format

from sidd.utils.common import get_datadir, get_logdir_root, get_vectorsdir
from sidd.data import AsbDataset


def load_vectors(name='embeddings'):
    with bz2.BZ2File(str(name) + '.pbz2', 'rb') as infile:
        result = pickle.load(infile)

    return result[0], result[1]


def save_vectors(values, labels, name='embeddings'):
    data = [values, labels]

    with bz2.BZ2File(str(name) + '.pbz2', 'wb') as f:
        pickle.dump(data, f, 4)


def save_embeddings(values, labels, name='embeddings'):
    return save_vectors(values, labels, get_vectorsdir(name))


def export_vectors(values, labels, name='embeddings'):
    header = ['Label', 'Embeddings']
    with open(get_datadir(name + '.csv'), 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(header)

        for i, (label) in enumerate(labels):
            label_str = ','.join(map(str, label))
            value_str = ','.join(map(str, values[i]))
            writer.writerow([i, label_str, value_str])


def calc_vectors(ds, model):
    ds_vectors = []
    ds_labels = []
    for images, labels in tqdm(ds):
        predictions = model(images)
        ds_vectors.extend(predictions.numpy().tolist())
        ds_labels.extend(labels.numpy().tolist())

    return np.array(ds_vectors, dtype='float32'), np.array(ds_labels, dtype='uint32')


def calc_vectors_fn(ds, fn, *args):
    ds_vectors = []
    ds_labels = []
    for image, label in tqdm(ds.as_numpy_iterator(), total=ds.cardinality().numpy()):
        vector = fn(image, *args)
        if vector is not None:
            ds_vectors.append(vector)
            ds_labels.append(label)

    return np.array(ds_vectors, dtype='float32'), np.array(ds_labels, dtype='uint32')


def evaluate_vectors(values, labels):
    total = len(values)
    match = 0
    missmatch = 0

    for i, (expected) in enumerate(labels):
        pred = np.argmax(values[i])
        if expected == pred:
            match += 1
        else:
            missmatch += 1

    return match / total


def project_embeddings(image_vectors, labels, name='projection'):
    root_dir = get_logdir_root()
    projection_name = name + '_' + str(time.strftime('%Y_%m_%d-%H_%M_%S'))
    projection_dir = root_dir.joinpath(projection_name)
    projection_dir.mkdir(parents=True, exist_ok=True)

    with (projection_dir / 'values.tsv').open('w') as fp:
        writer = csv.writer(fp, delimiter='\t')
        writer.writerows(image_vectors)

    with (projection_dir / 'metadata.tsv').open('w') as fp:
        for lbl in labels:
            fp.write(f'{lbl}\n')

    embeddings = tf.Variable(np.asarray(image_vectors), name='embeddings')
    ckpt = tf.train.Checkpoint(embeddings=embeddings)
    ckpt.save(str(projection_dir.joinpath('model.ckpt')))

    config = projector.ProjectorConfig()
    # load existing config if exists
    config_fpath = root_dir.joinpath(projector.metadata.PROJECTOR_FILENAME)
    if tf.io.gfile.exists(config_fpath):
        with tf.io.gfile.GFile(config_fpath, "r") as f:
            file_content = f.read()
        text_format.Merge(file_content, config)

    embedding = config.embeddings.add()
    embedding.tensor_name = projection_name
    embedding.metadata_path = projection_name + '/metadata.tsv'
    embedding.tensor_path = projection_name + '/values.tsv'
    projector.visualize_embeddings(root_dir, config)
