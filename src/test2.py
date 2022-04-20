# -*- coding: utf-8 -*-
"""Computing embeddings for siamese networks example.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1d10wqO7gADCU_jnpP7-VOnDpYyEsMjYx
"""
import sys
sys.path.append("..")

from src.data.embeddings import *
from utils.common import *
from utils.distance import *
from src.model.alexnet import AlexNetModel
from tensorflow.keras import layers, Model

from tqdm import tqdm
from pathlib import Path

"""## Load dataset and pretrained model backbone

### Model

Load the pretrained model from TF Hub. When building, we pass the input size that the model expects.
"""

# create model
alexnet = AlexNetModel()
alexnet.compile()
# alexnet.summary()

train_ds, test_ds, validation_ds = AlexNetModel.x_dataset()

# load weights
# alexnet.fit(train_ds, validation_data=test_ds)
# alexnet.save_weights(get_modeldir('cifar10_alexnet1304.h5'))
# alexnet.evaluate(validation_ds)
alexnet.load_weights(get_modeldir('cifar10_alexnet1304.h5'))

# image features
embedding_model = Model(inputs=alexnet.input, outputs=alexnet.layers[-2].output)
for layer in embedding_model.layers:
    layer.trainable = False
embedding_model.summary()

"""### Dataset

CIFAR 10 has shape 32x32 but the model expects 384x384, so we upsize the image (NOTE: this will likely lead to very bad performance, but it's because of CIFAR rather than the method itself. Consider using a dataset with higher image resolution; for first tests, stick to something available as tensorflow_dataset to speed up things a lot)
"""

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
images = np.concatenate([train_images, test_images])
labels = np.concatenate([train_labels, test_labels])
embedding_vds = tf.data.Dataset.from_tensor_slices((images, labels))
embedding_vds = embedding_vds.map(process_images_couple, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)

# Add batch and prefetch to dataset to speed up processing
BATCH_SIZE = 256
embedding_vds = embedding_vds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Dataset has keys "id" (that we ignore), "image" and "label".
# "image" has shape [BATCH_SIZE,32,32,3] and is an RGB uint8 image
# "label" has shape [BATCH_SIZE,1] and is an integer label (value between 0 and 9)

"""## Precompute embeddings for all dataset images

Since the network is frozen, to speed up training it's better to precalculate the image features for each image in the dataset and only use those values to train the siamese model.

For each image, we keep its label and the image features extracted by the model.
At the end, we save the computed embeddings as a Pandas dataframe, so they can be loaded back quickly without having to recompute them every time.

**NOTE**: Run this on a GPU-enabled runtime or it will take forever
"""

embeddings = embedding_model.predict(embedding_vds)
embedding_labels = np.concatenate([y for x, y in embedding_vds], axis=0)
embedding_labels = np.concatenate(embedding_labels).ravel()  # unwrap from single item array

NUM_CLASSES = np.unique(embedding_labels).shape[0]

"""# Siamese network training

Following this tutorial: https://keras.io/examples/vision/siamese_contrastive/

## Prepare the dataset

We already have the embeddings precomputed in `embeddings` and their matching `labels`. To train the siamese networks, we need to generate random pairs of embeddings, assigning as target `1` if the two come from the same class and `0` otherwise. 

In order to keep the training balanced, we can't simply select two random `(embedding, label)` tuples from the dataset, because this is heavily unbalanced towards the negative class. To keep thing simple, we'll randomly select two samples and then use `rejection_resample` to rebalance the classes.

**NOTE**: rejection resampling works only if the number of classes is reasonably low: with 10 classes there's a 90% probability that a sample will be rejected, it can get very inefficient very quickly if the number of classes is too great.
"""

# zip together embeddings and their labels, cache in memory (maybe not necessay or maybe faster this way), shuffle, repeat forever.
embeddings_ds = tf.data.Dataset.zip((
    tf.data.Dataset.from_tensor_slices(embeddings),
    tf.data.Dataset.from_tensor_slices(embedding_labels)
)).cache().shuffle(1000).repeat()

# change for triplet loss implementation


@tf.function
def make_label_for_pair(embeddings, labels):
    # embedding_1, label_1 = tuple_1
    # embedding_2, label_2 = tuple_2
    return (embeddings[0, :], embeddings[1, :]), tf.cast(labels[0] == labels[1], tf.float32)


# because of shuffling, we can take two adjacent tuples as a randomly matched pair
train_ds = embeddings_ds.window(2, drop_remainder=True)
train_ds = train_ds.flat_map(lambda w1, w2: tf.data.Dataset.zip((w1.batch(2), w2.batch(
    2))))  # see https://stackoverflow.com/questions/55429307/how-to-use-windows-created-by-the-dataset-window-method-in-tensorflow-2-0
# generate the target label depending on whether the labels match or not
train_ds = train_ds.map(make_label_for_pair, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
# resample to the desired distribution
# train_ds = train_ds.rejection_resample(lambda embs, target: tf.cast(target, tf.int32), [0.5, 0.5], initial_dist=[0.9, 0.1])
# train_ds = train_ds.map(lambda _, vals: vals) # discard the prepended "selected" class from the rejction resample, since we aleady have it available

"""## Model and loss definition

The `projection_model` is the part of the network that generates the final image vector (currently, a simple Dense layer with tanh activation, but it can be as complex as needed).

The `siamese` model is the one we train. It applies the projection model to two embeddings, calculates the euclidean distance between the two generated image vectors and calculates the contrastive loss.

As a note, [here](https://towardsdatascience.com/contrastive-loss-explaned-159f2d4a87ec) they mention that cosine distance is preferable to euclidean distance:

> in a large dimensional space, all points tend to be far apart by the euclidian measure. In higher dimensions, the angle between vectors is a more effective measure.

Note that, when using cosine distance, the margin needs to be reduced from its default value of 1 (see below).

__________________

### Contrastive Loss

$ Loss = Y*Dist(v_1,v_2)^2 + (1-Y)*max(margin-D,0)^2$

$Y$ is the GT target (1 if $v_1$ and $v_2$ belong to the same class, 0 otherwise). If images are from the same class, use the squared distance as loss (you want to push the distance to be close to 0 for same-class couples), otherwise keep the (squared) maximum between 0 and $margin - D$.

For different-class couples, the distance should be pushed to a high value. The **margin identifies a cone inside which vectors are considered the same**. For cosine distance, which has range [0,2], **1 is NOT an adequate value**).

**NOTE** In the loss implementation below, we calculate the mean of the two terms, though this should not actually be necessary (the minimizer value for the loss is the same whether the loss is divided by 2 or not).

"""

## Model hyperparters
EMBEDDING_VECTOR_DIMENSION = 4096
# EMBEDDING_VECTOR_DIMENSION = int(1280/2)
IMAGE_VECTOR_DIMENSIONS = 128
# IMAGE_VECTOR_DIMENSIONS = 3 # use for test visualization on tensorboard
ACTIVATION_FN = 'tanh'  # same as in paper
MARGIN = 0.05


## These functions are straight from the Keras tutorial linked above

# Provided two tensors t1 and t2
# Euclidean distance = sqrt(sum(square(t1-t2)))
def euclidean_distance(vects):
    """Find the Euclidean distance between two vectors.

    Arguments:
        vects: List containing two tensors of same length.

    Returns:
        Tensor containing euclidean distance
        (as floating point value) between vectors.
    """

    x, y = vects
    sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True)
    return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))


def cosine_distance(vects):
    """Find the Cosine distance between two vectors.

    Arguments:
        vects: List containing two tensors of same length.

    Returns:
        Tensor containing euclidean distance
        (as floating point value) between vectors.
    """
    # NOTE: Cosine_distance = 1 - cosine_similarity
    # Cosine distance is defined betwen [0,2] where 0 is vectors with the same direction and verse,
    # 1 is perpendicular vectors and 2 is opposite vectors
    cosine_similarity = tf.keras.layers.Dot(axes=1, normalize=True)(vects)
    return 1 - cosine_similarity


def loss(margin=1):
    """Provides 'constrastive_loss' an enclosing scope with variable 'margin'.

    Arguments:
        margin: Integer, defines the baseline for distance for which pairs
                should be classified as dissimilar. - (default is 1).

    Returns:
        'constrastive_loss' function with data ('margin') attached.
    """

    # Contrastive loss = mean( (1-true_value) * square(prediction) +
    #                         true_value * square( max(margin-prediction, 0) ))
    def contrastive_loss(y_true, y_pred):
        """Calculates the constrastive loss.

        Arguments:
            y_true: List of labels (1 for same-class pair, 0 for different-class), fp32.
            y_pred: List of predicted distances, fp32.

        Returns:
            A tensor containing constrastive loss as floating point value.
        """

        square_dist = tf.math.square(y_pred)
        margin_square = tf.math.square(tf.math.maximum(margin - (y_pred), 0))
        return tf.math.reduce_mean(
            (1 - y_true) * square_dist + (y_true) * margin_square
        )

    return contrastive_loss


emb_input_1 = layers.Input(EMBEDDING_VECTOR_DIMENSION)
emb_input_2 = layers.Input(EMBEDDING_VECTOR_DIMENSION)

# todo Add more layers here

# projection model is the one to use for queries (put in a sequence after the embedding-generator model above)
projection_model = tf.keras.models.Sequential([
    layers.Dense(IMAGE_VECTOR_DIMENSIONS, activation=ACTIVATION_FN, input_shape=(EMBEDDING_VECTOR_DIMENSION,))
    # layers.Dense(128, activation='relu', input_shape=(EMBEDDING_VECTOR_DIMENSION,)),
    # layers.Dense(IMAGE_VECTOR_DIMENSIONS, activation=None)
    # relu on activation, max
])

v1 = projection_model(emb_input_1)
v2 = projection_model(emb_input_2)

computed_distance = layers.Lambda(cosine_distance)([v1, v2])
# siamese is the model we train
siamese = Model(inputs=[emb_input_1, emb_input_2], outputs=computed_distance)

## Training hyperparameters (values selected randomly at the moment, would be easy to set up hyperparameter tuning wth Keras Tuner)
## We have 128 pairs for each epoch, thus in total we will have 128 x 2 x 1000 images to give to the siamese
TRAIN_BATCH_SIZE = 128
STEPS_PER_EPOCH = 1000
NUM_EPOCHS = 3

# TODO: If there's a need to adapt the learning rate, explicitly create the optimizer instance here and pass it into compile
siamese.compile(loss=loss(margin=MARGIN), optimizer="RMSprop")
siamese.summary()

"""Select Projector interface for Tensorboard"""

# Commented out IPython magic to ensure Python compatibility.
# %load_ext tensorboard
# %tensorboard --logdir=logs

callbacks = [
    tf.keras.callbacks.TensorBoard(get_logdir("inference/fit"), profile_batch=5)
]

# TODO: Would be good to have a validation dataset too.

ds = train_ds.batch(TRAIN_BATCH_SIZE)  # .prefetch(tf.data.AUTOTUNE)
history = siamese.fit(
    ds,
    epochs=NUM_EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    callbacks=callbacks,
    class_weight={0: 1 / NUM_CLASSES, 1: (NUM_CLASSES - 1) / NUM_CLASSES}
)

# Build full inference model (from image to image vector):

im_input = embedding_model.input
embedding = embedding_model(im_input)
image_vector = projection_model(embedding)
inference_model = Model(inputs=im_input, outputs=image_vector)

inference_model.save(get_modeldir('inference_model.tf'), save_format='tf', include_optimizer=False)

"""# Validation

To validate the model, we load the validation chunk of the dataset and we feed it into the network. We don't need to repeat the preprocessing steps done to the dataset, because the preprocessing is embedded in the inference model by the `Rescaling` and `Resizing` layers we added above.

____________

## Visualizing embeddings in TensorBoard

In `metadata.tsv` file we list the labels in the same order as they appear in the embeddings list.
We write out the embeddings list as a tf.Variable initialized to the embeddings values, using TensorBoard's writers to specify the metadata file to use and the name of the tensor to display.

Additionally, in the specification of ProjectorConfig's proto message, there is the possibility to pass the values as a second .tsv file (`values.tsv`) instead than having them loaded from the checkpoint file.

I don't know which values are getting loaded at the moment, but since it works I won't change it further and keep both the .tsv and the checkpointed values.

(See https://stackoverflow.com/a/57230031/3214872)
"""

print('visualization')

def write_embeddings_for_tensorboard(image_vectors: list, labels: list, root_dir: Path):
    import csv
    from tensorboard.plugins import projector
    root_dir.mkdir(parents=True, exist_ok=True)
    with (root_dir / 'values.tsv').open('w') as fp:
        writer = csv.writer(fp, delimiter='\t')
        writer.writerows(image_vectors)

    with (root_dir / 'metadata.tsv').open('w') as fp:
        for lbl in labels:
            fp.write(f'{lbl}\n')

    image_vectors = np.asarray(image_vectors)
    embeddings = tf.Variable(image_vectors, name='embeddings')
    CHECKPOINT_FILE = str(root_dir / 'model.ckpt')
    ckpt = tf.train.Checkpoint(embeddings=embeddings)
    ckpt.save(CHECKPOINT_FILE)

    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = "embeddings/.ATTRIBUTES/VARIABLE_VALUE"
    embedding.metadata_path = 'metadata.tsv'
    embedding.tensor_path = 'values.tsv'
    projector.visualize_embeddings(root_dir, config)


inference_model = tf.keras.models.load_model(get_modeldir('inference_model.tf'), compile=False)

# NUM_SAMPLES_TO_DISPLAY = 10000
NUM_SAMPLES_TO_DISPLAY = 3000
LOG_DIR = Path('../logs/logs_projection0413')

LOG_DIR.mkdir(exist_ok=True, parents=True)

# compute embeddings of the images and their labels, store them in a tsv file for visualization
image_vectors = inference_model.predict(embedding_vds)
labels = embedding_labels
# for feats_batch in tqdm(embedding_vds):
#     ims = feats_batch['image']
#     lbls = feats_batch['label'].numpy()
#     embs = inference_model(ims).numpy()
#     image_vectors.extend(embs.tolist())
#     labels.extend(lbls.tolist())

write_embeddings_for_tensorboard(image_vectors, labels, LOG_DIR)

# # Do the same with some of the training data, just to see if it works with that
# ds = embeddings_ds.take(NUM_SAMPLES_TO_DISPLAY).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
# _image_vectors = []
# _labels = []
# for feats_batch in tqdm(ds):
#     ims, lbls = feats_batch
#     ims = ims.numpy()
#     lbls = lbls.numpy()
#     embs = projection_model(ims).numpy()
#     _image_vectors.extend(embs.tolist())
#     _labels.extend(lbls.tolist())
# write_embeddings_for_tensorboard(_image_vectors, _labels, LOG_DIR / 'train')