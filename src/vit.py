import sys

sys.path.append("..")

import numpy as np
import tensorflow as tf
from src.utils.embeddings import project_embeddings
from src.utils.common import get_modeldir, get_datadir
from src.model.vit import VitModel, MODEL_INPUT_SIZE
from src.model.siamese import SiameseModel

import tensorflow_datasets as tfds
import pandas as pd

from tqdm import tqdm
from pathlib import Path

model_name = 'cifar10_vit'
embeddings_name = model_name + '_embeddings'

embedding_model = VitModel()
embedding_model.summary()

# DATASET_NAME = 'cats_vs_dogs'
DATASET_NAME = 'cifar10'
# DATASET_NAME = 'cars196'

# NOTE: For cars196 & other datasets with many classes, the rejection resampling
#      used to balance the positive and negative classes does NOT work anymore! (the input pipeline chokes)
#      Need to find a better solution!
# -> FIX: Using class weights based on the number of labels in the original dataset seems to work perfectly well (and training speed improves greatly too)

# Load dataset in a form already consumable by Tensorflow
ds = tfds.load(DATASET_NAME, split='train')


# Resize images to the model's input size and normalize to [0.0, 1.0] as per the
# expected image input signature: https://www.tensorflow.org/hub/common_signatures/images#input
def resize_and_normalize(features):
    return {
        # 'id': features['id'],
        'label': features['label'],
        'image': (tf.image.resize(tf.image.convert_image_dtype(features['image'], tf.float32),
                                  MODEL_INPUT_SIZE[1:3]) - 0.5) * 2  # ViT requires images in range [-1,1]
    }


ds = ds.map(resize_and_normalize, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)

# Add batch and prefetch to dataset to speed up processing
BATCH_SIZE = 256
batched_ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Dataset has keys "id" (that we ignore), "image" and "label".
# "image" has shape [BATCH_SIZE,32,32,3] and is an RGB uint8 image
# "label" has shape [BATCH_SIZE,1] and is an integer label (value between 0 and 9)

# Naming schema: <dataset_name>-<dataset_split>.<model-name>.embeddings.pickle
DST_FNAME = get_datadir('vit_s16_fe.embeddings.pkl')

if Path(DST_FNAME).exists():
    # When you need to use the embeddings, upload the file (or store it on Drive and mount your drive folder in Colab), then run:
    df = pd.read_pickle(DST_FNAME)  # adapt the path as needed
    embeddings = np.array(df.embedding.values.tolist())
    labels = df.label.values
else:
    embeddings = []
    labels = []
    for features_batch in tqdm(batched_ds):
        embeddings.append(embedding_model(features_batch['image']).numpy())
        labels.append(features_batch['label'].numpy())

    embeddings = np.concatenate(embeddings)
    labels = np.concatenate(labels)

    # Store the precompued values to disk
    df = pd.DataFrame({'embedding': embeddings.tolist(), 'label': labels})
    df.to_pickle(DST_FNAME)
    # Download the generated file to store the calculated embeddings.

NUM_CLASSES = np.unique(labels).shape[0]

# siamese is the model we train
siamese = SiameseModel(embedding_vector_dimension=384, image_vector_dimensions=512)
siamese.compile(loss_margin=0.005, optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
siamese.summary()

## Training hyperparameters (values selected randomly at the moment, would be easy to set up hyperparameter tuning wth Keras Tuner)
## We have 128 pairs for each epoch, thus in total we will have 128 x 2 x 1000 images to give to the siamese
NUM_EPOCHS = 10
TRAIN_BATCH_SIZE = 128
STEPS_PER_EPOCH = 3000

ds = SiameseModel.prepare_dataset(embeddings, labels)
history = siamese.fit(
    ds,
    epochs=NUM_EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    class_weight={0: 1 / NUM_CLASSES, 1: (NUM_CLASSES - 1) / NUM_CLASSES}
)

# Build full inference model (from image to image vector):
inference_model = siamese.get_inference_model(embedding_model)
inference_model.save(get_modeldir(model_name + '_inference.tf'), save_format='tf', include_optimizer=False)

# inference_model = tf.keras.models.load_model(get_modeldir(model_name + '_inference.tf'), compile=False)

print('visualization')
# compute vectors of the images and their labels, store them in a tsv file for visualization
image_vectors = siamese.get_projection_model().predict(embeddings)
project_embeddings(image_vectors, labels, model_name)
