# This script evaluates SiameseCNN on evaluation dataset.
# Required parameters:
#   --model: CNN model name
#   --dataset: Dataset name (used to load weight)
#   --eval-dataset: Evaluation dataset name
#   --seed: A seed (for loading weights or training)

import time
import logging as log
import tensorflow as tf
from tensorflow.keras import Model

from cnn_base import evaluate_combined, load_cnn, load_embeddings
from sidd import SiameseCliParams
from sidd.data import AbsDataset
from sidd.utils.common import get_modeldir


def load_siamesecnn(params: SiameseCliParams, ds: AbsDataset, train = True) -> Model:
    model_file = get_modeldir('siamese_inference_' + params.siamesecnn_name + '.tf')
    if model_file.exists():
        return tf.keras.models.load_model(model_file)
    elif train:
        print('Inference model does not exist, training...')
        emb_model = load_cnn(params, ds).get_embedding_model()
        emb_model.summary()

        vectors, labels = load_embeddings(emb_model, ds, params.seed)

        emb_ds = params.get_siamese_class().prepare_dataset(vectors, labels)
        siamese_model = params.get_siamese_class()(embedding_model=emb_model, image_vector_dimensions=params.dimensions,
                                            loss_margin=params.margin, fit_epochs=params.epochs, basename=params.siamesecnn_name)
        siamese_model.compile(loss=params.get_loss_class())
        siamese_model.summary()

        start = time.time()
        siamese_model.fit(emb_ds, num_classes=ds.num_classes)
        log.info('Siamese model %s trained in %ss', params.siamesecnn_name, time.time() - start)
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

    inference_model = load_siamesecnn()

    if params.eval_dataset is not None:
        eval_ds = params.get_eval_dataset( # Evaluation dataset
            image_size=params.get_model_class().get_target_shape(),
            map_fn=params.get_model_class().preprocess_input
        )

        evaluate_combined(params, inference_model, eval_ds)

    print('Done!\n')
