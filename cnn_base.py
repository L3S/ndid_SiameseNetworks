# This script evaluates a trained CNN model (or trains CNN model if weights are not available) on the evaluation dataset.
# Required parameters:
#   --model: CNN model name
#   --dataset: Dataset name (used to load weight or train model)
#   --eval-dataset: Evaluation dataset name
#   --seed: A seed (for loading weights or training)

import time
import numpy as np
import logging as log
import tensorflow as tf
from tensorflow.keras import Model

from nnfaiss import knn_search, range_search
from sidd import SiameseCliParams
from sidd.data import AbsDataset
from sidd.utils.common import get_modeldir, get_datadir
from sidd.utils.embeddings import project_embeddings, calc_vectors, save_vectors, save_embeddings, load_vectors

tf.get_logger().setLevel('INFO')
log.basicConfig(filename="logfile.log", level=log.INFO, format='%(asctime)s %(message)s')


def train_cnn(model: Model, ds: AbsDataset):
    start = time.time()
    model.fit(ds.get_train(), validation_data=ds.get_val())
    log.info('Model %s trained in %ss', model.name, time.time() - start)

    if ds.get_test() is not None:
        print('Model trained, evaluating...')
        model.evaluate(ds.get_test())


def load_cnn(params: SiameseCliParams, ds: AbsDataset, train = True) -> Model:
    model = params.get_model(train_size=len(ds.get_train()), num_classes=ds.num_classes, weights=params.weights)
    model.compile()

    if params.weights == 'imagenet':
        print('Alexnet model loaded, skipping training.')
    else:
        model_file = get_modeldir(params.cnn_name + '.h5')
        if model_file.exists():
            print('Loading model weights from %s', model_file)
            model.load_weights(model_file)
            print('Model weights loaded.')
        elif train:
            print('Model weights do not exist, training...')
            train_cnn(model, ds)
            model.save_weights(model_file)
        else:
            print('Model weights do not exist. Exiting.')
            exit(0)

    return model


def evaluate_combined(params: SiameseCliParams, model: Model, ds: AbsDataset):
    print('Computing ' + ds.name + ' vectors...')
    eval_vectors, eval_labels = calc_vectors(ds.get_combined(), model)

    if params.save_vectors:
        save_embeddings(eval_vectors, eval_labels, 'eval_' + ds.name + '_' +  model.name + '_vectors')

    if params.compute_stats:
        knn_search.compute_and_save(eval_vectors, eval_labels, 'eval_' + ds.name + '_' +  model.name + '_knn', True)
        range_search.compute_and_save(eval_vectors, eval_labels, 'eval_' + ds.name + '_' +  model.name + '_range', True)

    if params.project_vectors:
        project_embeddings(eval_vectors, eval_labels, 'eval_' + ds.name + '_' +  model.name)


def load_embeddings(model: Model, ds: AbsDataset, seed: str) -> tuple[np.ndarray, np.ndarray]:
    save_file = get_datadir(model.name + '_' + ds.name + '_' + seed)
    if save_file.exists():
        return load_vectors(save_file)
    else:
        print('Calculating embeddings...')
        vectors, labels = calc_vectors(ds.get_train(), model)
        save_vectors(vectors, labels, save_file)
        return vectors, labels


if __name__ == "__main__":
    params = SiameseCliParams.parse()

    dataset = params.get_dataset( # Model train dataset
        image_size=params.get_model_class().get_target_shape(),
        map_fn=params.get_model_class().preprocess_input
    )

    emb_model = load_cnn(params, dataset).get_embedding_model()
    emb_model.summary()

    if params.eval_dataset is not None:
        eval_ds = params.get_eval_dataset( # Evaluation dataset
            image_size=params.get_model_class().get_target_shape(),
            map_fn=params.get_model_class().preprocess_input
        )

        evaluate_combined(params, emb_model, eval_ds)

    print('Done!\n')
