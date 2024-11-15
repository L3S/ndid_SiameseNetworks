# This script evaluates SiameseCNN on evaluation dataset.
# Required parameters:
#   --cnn-model: CNN model
#   --cnn-dataset: CNN Dataset name (used to load weight)
#   --dataset: Datased on which SiameseCNN was trained
#   --eval-dataset: Evaluation dataset name
#   --seed: A seed (for loading weights or training)

import time
import logging as log
import tensorflow as tf
from functools import cache

from cnn_base import evaluate, load_cnn, load_embeddings
from sidd import SiameseCliParams
from sidd.data import AbsDataset
from sidd.utils.common import get_modeldir, get_vectorsdir


def siamese_name(params: SiameseCliParams, margin, dimensions, epochs) -> str:
    return params.core_name + '_d' + str(dimensions) + '_m' + str(margin) + '_s' + str(epochs * 100) + '_' + params.loss + '_' + params.seed


@cache
def get_embedded(params: SiameseCliParams, ds: AbsDataset) -> tuple[tf.keras.Model, tf.data.Dataset]:
    start = time.time()
    emb_model = load_cnn(params).get_embedding_model()
    emb_model.summary()

    vectors, labels = load_embeddings(emb_model, ds.get_train(), ds.name, params.seed)
    emb_ds = params.get_siamese_class().prepare_dataset(vectors, labels)
    print('Embeddings calculated in %ss', time.time() - start)
    return emb_model, emb_ds


def load_siamesecnn(params: SiameseCliParams, ds: AbsDataset, train = True, margin=None, dimensions=None, epochs=None) -> tf.keras.Model:
    if margin is None:
        margin = params.margin[0]
    if dimensions is None:
        dimensions = params.dimensions[0]
    if epochs is None:
        epochs = params.epochs[0]

    siamesecnn_name = siamese_name(params, margin, dimensions, epochs)

    print('Loading SiameseCNN with margin %s, dimensions %s, epochs %s...', margin, dimensions, epochs)
    model_file = get_modeldir('siamese_inference_' + siamesecnn_name + '.tf')
    if model_file.exists():
        return tf.keras.models.load_model(model_file)
    elif train:
        print('Inference model does not exist, training...')
        start = time.time()
        emb_model, emb_ds = get_embedded(params, ds)
        siamese_model = params.get_siamese_class()(embedding_model=emb_model, basename=siamesecnn_name,
                                                   image_vector_dimensions=dimensions, loss_margin=margin, fit_epochs=epochs)
        siamese_model.compile(loss=params.get_loss_class(), optimizer=tf.keras.optimizers.RMSprop())
        siamese_model.summary()

        start_fit = time.time()
        siamese_model.fit(emb_ds, num_classes=ds.num_classes)
        print('Siamese model %s loaded & trained in %ss', siamesecnn_name, time.time() - start)
        log.info('Siamese model %s trained in %ss', siamesecnn_name, time.time() - start_fit)
        siamese_model.inference_model.save(model_file)
        return siamese_model.inference_model
    else:
        print('Model weights do not exist. Exiting.')
        exit(0)


if __name__ == "__main__":
    params = SiameseCliParams.parse()

    dataset = params.get_dataset( # Model train dataset
        image_size=params.get_model_class().get_target_shape(),
        map_fn=params.get_model_class().preprocess_input
    )

    for margin in params.margin:
        for epochs in params.epochs:
            for dimensions in params.dimensions:
                inference_model = load_siamesecnn(params, dataset, margin=margin, dimensions=dimensions, epochs=epochs)

                vectors_file = get_vectorsdir('eval_' + dataset.name + '_' +  inference_model.name + '_vectors' + '.pbz2')
                if params.eval_dataset is not None:
                    print('Evaluating on combined dataset...')
                    eval_ds = params.get_eval_dataset( # Evaluation dataset
                        image_size=params.get_model_class().get_target_shape(),
                        map_fn=params.get_model_class().preprocess_input
                    )

                    vectors_file = get_vectorsdir('eval_' + eval_ds.name + '-full' + '_' +  inference_model.name + '_vectors' + '.pbz2')
                    if not vectors_file.exists():
                        evaluate(params, inference_model, eval_ds.get_combined(), eval_ds.name + '-full')
                elif not vectors_file.exists():
                    print('Evaluating on test dataset...')
                    evaluate(params, inference_model, dataset.get_test(), dataset.name)

                print('Done!\n')
