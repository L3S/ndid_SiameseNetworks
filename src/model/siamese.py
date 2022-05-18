from src.utils.common import *
import tensorflow_addons as tfa
from src.utils.distance import cosine_distance, euclidean_distance
from tensorflow.keras import layers, callbacks, Model

tensorboard_cb = callbacks.TensorBoard(get_logdir('siamese/fit'))

EMBEDDING_VECTOR_DIMENSION = 4096
IMAGE_VECTOR_DIMENSIONS = 3  # use for test visualization on tensorboard
ACTIVATION_FN = 'tanh'  # same as in paper
MARGIN = 0.5

NUM_EPOCHS = 3
TRAIN_BATCH_SIZE = 128
STEPS_PER_EPOCH = 100  # 1000


@tf.function
def make_label_for_pair(embeddings, labels):
    # embedding_1, label_1 = tuple_1
    # embedding_2, label_2 = tuple_2
    return (embeddings[0, :], embeddings[1, :]), tf.cast(labels[0] == labels[1], tf.float32)


class SiameseModel(Model):
    """ Filippo's Siamese model

    The `projection_model` is the part of the network that generates the final image vector (currently, a simple Dense layer with tanh activation, but it can be as complex as needed).

    The `siamese` model is the one we train. It applies the projection model to two embeddings, calculates the euclidean distance between the two generated image vectors and calculates the contrastive loss.

    As a note, [here](https://towardsdatascience.com/contrastive-loss-explaned-159f2d4a87ec) they mention that cosine distance is preferable to euclidean distance:

    > in a large dimensional space, all points tend to be far apart by the euclidian measure. In higher dimensions, the angle between vectors is a more effective measure.

    Note that, when using cosine distance, the margin needs to be reduced from its default value of 1 (see below).
    """

    def __init__(self, embedding_vector_dimension=EMBEDDING_VECTOR_DIMENSION, image_vector_dimensions=IMAGE_VECTOR_DIMENSIONS):
        super().__init__()

        emb_input_1 = layers.Input(embedding_vector_dimension)
        emb_input_2 = layers.Input(embedding_vector_dimension)

        # projection model is the one to use for queries (put in a sequence after the embedding-generator model above)
        self.projection_model = tf.keras.models.Sequential([
            # layers.Dense(image_vector_dimensions, activation=ACTIVATION_FN, input_shape=(embedding_vector_dimension,))
            layers.Dense(128, activation='relu', input_shape=(embedding_vector_dimension,)),
            layers.Dense(image_vector_dimensions, activation=None),
            layers.Lambda(lambda x: tf.keras.backend.l2_normalize(x, axis=1)),
        ])

        v1 = self.projection_model(emb_input_1)
        v2 = self.projection_model(emb_input_2)
        computed_distance = layers.Lambda(cosine_distance)([v1, v2])
        # computed_distance = layers.Lambda(euclidean_distance)([v1, v2])

        super(SiameseModel, self).__init__(inputs=[emb_input_1, emb_input_2], outputs=computed_distance)

    def get_projection_model(self):
        """ Projection model is a model from embeddings to image vector """
        return self.projection_model

    def get_inference_model(self, embedding_model):
        """ Inference model is a model from image to image vector """
        im_input = embedding_model.input
        embedding = embedding_model(im_input)
        image_vector = self.projection_model(embedding)
        return Model(inputs=im_input, outputs=image_vector)

    def compile(self,
                optimizer=tf.keras.optimizers.RMSprop(),
                loss_margin=MARGIN,
                **kwargs):
        super().compile(optimizer=optimizer, loss=tfa.losses.ContrastiveLoss(margin=loss_margin), **kwargs)

    def fit(self, x=None, y=None, batch_size=None, epochs=NUM_EPOCHS, steps_per_epoch=STEPS_PER_EPOCH, callbacks=[tensorboard_cb], **kwargs):
        return super().fit(x=x, y=y, batch_size=batch_size, epochs=epochs, steps_per_epoch=steps_per_epoch, callbacks=callbacks, **kwargs)

    @staticmethod
    def prepare_dataset(embeddings, labels):
        """
        We already have the embeddings precomputed in `embeddings` and their matching `labels`. To train the siamese networks, we need to generate random pairs of embeddings, assigning as target `1` if the two come from the same class and `0` otherwise.

        In order to keep the training balanced, we can't simply select two random `(embedding, label)` tuples from the dataset, because this is heavily unbalanced towards the negative class. To keep thing simple, we'll randomly select two samples and then use `rejection_resample` to rebalance the classes.

        **NOTE**: rejection resampling works only if the number of classes is reasonably low: with 10 classes there's a 90% probability that a sample will be rejected, it can get very inefficient very quickly if the number of classes is too great.
        """

        # zip together embeddings and their labels, cache in memory (maybe not necessary or maybe faster this way), shuffle, repeat forever.
        embeddings_ds = tf.data.Dataset.zip((
            tf.data.Dataset.from_tensor_slices(embeddings),
            tf.data.Dataset.from_tensor_slices(labels)
        )).cache().shuffle(1000).repeat()

        # TODO: change for triplet loss implementation
        # because of shuffling, we can take two adjacent tuples as a randomly matched pair
        train_ds = embeddings_ds.window(2, drop_remainder=True)
        train_ds = train_ds.flat_map(lambda w1, w2: tf.data.Dataset.zip((w1.batch(2), w2.batch(2))))  # see https://stackoverflow.com/questions/55429307/how-to-use-windows-created-by-the-dataset-window-method-in-tensorflow-2-0
        # generate the target label depending on whether the labels match or not
        train_ds = train_ds.map(make_label_for_pair, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
        # resample to the desired distribution
        # train_ds = train_ds.rejection_resample(lambda embs, target: tf.cast(target, tf.int32), [0.5, 0.5], initial_dist=[0.9, 0.1])
        # train_ds = train_ds.map(lambda _, vals: vals) # discard the prepended "selected" class from the rejction resample, since we aleady have it available
        train_ds = train_ds.batch(TRAIN_BATCH_SIZE)  # .prefetch(tf.data.AUTOTUNE)
        return train_ds
