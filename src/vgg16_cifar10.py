import sys
sys.path.append("..")

import numpy as np
from src.data.cifar10 import *
from src.data.embeddings import *
from utils.common import *
from tensorflow.keras import layers, Model
from pathlib import Path

model_name = 'cifar10_vgg16'
embeddings_name = model_name + '_embeddings'

train_ds, test_ds = load_dataset()

train_ds_size = tf.data.experimental.cardinality(train_ds).numpy()
fit_ds = train_ds.skip(train_ds_size / 10)
val_ds = train_ds.take(train_ds_size / 10)

fit_ds_size = tf.data.experimental.cardinality(fit_ds).numpy()
val_ds_size = tf.data.experimental.cardinality(val_ds).numpy()
test_ds_size = tf.data.experimental.cardinality(test_ds).numpy()
print("Training data size:", tf.data.experimental.cardinality(fit_ds).numpy())
print("Validation data size:", tf.data.experimental.cardinality(val_ds).numpy())
print("Evaluation data size:", tf.data.experimental.cardinality(test_ds).numpy())

# load model
# model = models.load_model(get_modeldir(model_name + '.tf'))

# create model
model = tf.keras.applications.VGG16(
    include_top=True,
    input_shape=(32, 32, 3),
    weights=None,
    classes=10
)

PRETRAIN_EPOCHS = 20
PRETRAIN_TOTAL_STEPS = PRETRAIN_EPOCHS * len(fit_ds)

loss = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.RMSprop(tf.keras.optimizers.schedules.CosineDecay(1e-3, PRETRAIN_TOTAL_STEPS))

model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
model.summary()

# load weights
model.load_weights(get_modeldir(model_name + '.h5'))

# train
# model.fit(fit_ds, epochs=PRETRAIN_EPOCHS, validation_data=val_ds)

# save
# model.save_weights(get_modeldir(model_name + '.h5'))
# model.save(get_modeldir(model_name + '.tf'))

# evaluate
print('evaluating...')
model.evaluate(test_ds)

for layer in model.layers:
    layer.trainable = False

# save embeddings
embedding_model = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)
embedding_model.summary()

embedding_vds = train_ds.concatenate(test_ds)

print('calculating embeddings...')
# embeddings = embedding_model.predict(embedding_vds)
# embedding_labels = np.concatenate([y for x, y in embedding_vds], axis=0)

# save_embeddings(embeddings, embedding_labels, embeddings_name)
embeddings, embedding_labels = load_embeddings(embeddings_name)
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
IMAGE_VECTOR_DIMENSIONS = 3
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
    # layers.Dense(IMAGE_VECTOR_DIMENSIONS, activation=ACTIVATION_FN, input_shape=(EMBEDDING_VECTOR_DIMENSION,))
    layers.Dense(128, activation='relu', input_shape=(EMBEDDING_VECTOR_DIMENSION,)),
    layers.Dense(IMAGE_VECTOR_DIMENSIONS, activation=None)
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

inference_model.save(get_modeldir(model_name + '_inference.tf'), save_format='tf', include_optimizer=False)

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


# inference_model = tf.keras.models.load_model(get_modeldir(model_name + '_inference.tf'), compile=False)

# NUM_SAMPLES_TO_DISPLAY = 10000
NUM_SAMPLES_TO_DISPLAY = 3000
LOG_DIR = Path('../logs/logs_projection0428')

LOG_DIR.mkdir(exist_ok=True, parents=True)

# compute embeddings of the images and their labels, store them in a tsv file for visualization
image_vectors = inference_model.predict(embedding_vds)
labels = embedding_labels

write_embeddings_for_tensorboard(image_vectors, labels, LOG_DIR)
