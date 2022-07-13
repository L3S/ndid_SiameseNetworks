import csv
import time
import bz2
import pickle
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from tensorboard.plugins import projector
from google.protobuf import text_format

from src.utils.common import get_datadir, get_modeldir, get_logdir_root
from src.data import AsbDataset


def _save_vectors_path(values, labels, path):
    data = [values, labels]

    with bz2.BZ2File(path, 'wb') as f:
        pickle.dump(data, f, 4)


def _load_vectors_path(path):
    with bz2.BZ2File(path, 'rb') as infile:
        result = pickle.load(infile)

    return result[0], result[1]


def load_vectors(name='embeddings'):
    return _load_vectors_path(get_datadir(name + '.pbz2'))


def save_vectors(values, labels, name='embeddings'):
    return _save_vectors_path(values, labels, get_datadir(name + '.pbz2'))


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

    return np.array(ds_vectors, dtype='float32'), np.array(ds_labels, dtype='uint8')


def calc_vectors_fn(ds, fn, *args):
    ds_vectors = []
    ds_labels = []
    for image, label in tqdm(ds.as_numpy_iterator(), total=ds.cardinality().numpy()):
        vector = fn(image, *args)
        if vector is not None:
            ds_vectors.append(vector)
            ds_labels.append(label)

    return np.array(ds_vectors, dtype='float32'), np.array(ds_labels, dtype='uint8')


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


def load_weights_of(model: tf.keras.Model, dataset: AsbDataset):
    model_file = get_modeldir(model.name + '_' + dataset.name + '.h5')

    if model_file.exists():
        model.load_weights(model_file)
    else:
        print('Model weights do not exist, training...')
        model.fit(dataset.get_train(), validation_data=dataset.get_val())
        model.save_weights(model_file)

        print('Model trained, evaluating...')
        model.evaluate(dataset.get_test())


def get_embeddings_of(model: tf.keras.Model, dataset: AsbDataset):
    embedding_file = get_datadir(model.name + '_' + dataset.name + '.pbz2')

    if embedding_file.exists():
        return _load_vectors_path(embedding_file)
    else:
        print('calculating vectors...')
        emb_vectors, emb_labels = calc_vectors(dataset.get_combined(), model)
        _save_vectors_path(emb_vectors, emb_labels, embedding_file)
        return emb_vectors, emb_labels
