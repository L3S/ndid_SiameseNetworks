import csv
import time
import _pickle as pickle
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from pathlib import Path
from tensorboard.plugins import projector
from google.protobuf import text_format

from src.utils.common import get_datadir, get_modeldir, get_logdir_root
from src.data.base import BaseDataset


def save_embeddings(values, labels, name='embeddings'):
    data = [values, labels]

    with open(get_datadir(name + '.pkl'), 'wb') as outfile:
        pickle.dump(data, outfile, -1)


def load_embeddings(name='embeddings'):
    with open(get_datadir(name + '.pkl'), 'rb') as infile:
        result = pickle.load(infile)

    return result[0], result[1]


def export_embeddings(values, labels, name='embeddings'):
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
    for ds_batch in tqdm(ds):
        images, labels = ds_batch
        predictions = model(images)
        ds_vectors.extend(predictions.numpy().tolist())
        ds_labels.extend(labels.numpy().tolist())

    return np.array(ds_vectors), np.array(ds_labels)


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
    root_dir = Path(get_logdir_root())
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


def load_weights_of(model: tf.keras.Model, dataset: BaseDataset):
    model_file = get_modeldir(model.name + '_' + dataset.name + '.h5')

    if Path(model_file).exists():
        model.load_weights(model_file)
    else:
        print('Model weights do not exist, training...')
        model.fit(dataset.get_train(), validation_data=dataset.get_val())
        model.save_weights(model_file)

        print('Model trained, evaluating...')
        model.evaluate(dataset.get_test())


def get_embeddings_of(model: tf.keras.Model, dataset: BaseDataset):
    embedding_file = get_datadir(model.name + '_' + dataset.name + '.pkl')

    if Path(embedding_file).exists():
        with open(embedding_file, 'rb') as infile:
            emb_vectors, emb_labels = pickle.load(infile)
        return emb_vectors, emb_labels
    else:
        print('calculating vectors...')
        ds_vectors = []
        ds_labels = []
        for ds_batch in tqdm(dataset.get_combined()):
            images, labels = ds_batch
            predictions = model(images)
            ds_vectors.extend(predictions.numpy().tolist())
            ds_labels.extend(labels.numpy().tolist())

        emb_vectors = np.array(ds_vectors)
        emb_labels = np.array(ds_labels)

        data = [emb_vectors, emb_labels]
        with open(embedding_file, 'wb') as outfile:
            pickle.dump(data, outfile, -1)

        return emb_vectors, emb_labels
