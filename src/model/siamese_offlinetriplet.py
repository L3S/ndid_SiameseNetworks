from src.utils.common import *
import tensorflow_addons as tfa
from src.utils.distance import cosine_distance, euclidean_distance
from tensorflow.keras import layers, callbacks, Model
from src.utils.losses import OfflineTripletLoss

tensorboard_cb = callbacks.TensorBoard(get_logdir('siamese/fit'), histogram_freq=1)

EMBEDDING_VECTOR_DIMENSION = 4096
IMAGE_VECTOR_DIMENSIONS = 3  # use for test visualization on tensorboard
ACTIVATION_FN = 'tanh'  # same as in paper
DEFAULT_MARGIN = 0.5

NUM_EPOCHS = 10
TRAIN_BATCH_SIZE = 64
STEPS_PER_EPOCH = 100

from src.utils.dataset_stats import getLabelStats, getEmbStats
import numpy as np
from src.utils.tuple import shuffle_arrays


@tf.function
def make_label_for_triplet(emb1, emb2, emb3):
    return ((emb1, emb2, emb3)), tf.cast(0, tf.float32)

class SiameseOfflineTripletModel(Model):
    """ Filippo's Siamese model

    The `projection_model` is the part of the network that generates the final image vector (currently, a simple Dense layer with tanh activation, but it can be as complex as needed).

    The `siamese` model is the one we train. It applies the projection model to two embeddings, calculates the euclidean distance between the two generated image vectors and calculates the contrastive loss.

    As a note, [here](https://towardsdatascience.com/contrastive-loss-explaned-159f2d4a87ec) they mention that cosine distance is preferable to euclidean distance:

    > in a large dimensional space, all points tend to be far apart by the euclidian measure. In higher dimensions, the angle between vectors is a more effective measure.

    Note that, when using cosine distance, the margin needs to be reduced from its default value of 1 (see below).
    """

    def __init__(self, embedding_model, image_vector_dimensions=IMAGE_VECTOR_DIMENSIONS):
        super().__init__()


        self.embedding_model = embedding_model
        self.embedding_vector_dimension = embedding_model.output_shape[1]
        self.image_vector_dimensions = image_vector_dimensions

        emb_input_1 = layers.Input(self.embedding_vector_dimension)
        emb_input_2 = layers.Input(self.embedding_vector_dimension)

        emb_input_3 = layers.Input(self.embedding_vector_dimension)

        """ Projection model is a model from embeddings to image vector """
        # projection model is the one to use for queries (put in a sequence after the embedding-generator model above)
        self.projection_model = tf.keras.models.Sequential([
            # layers.Dense(image_vector_dimensions, activation=ACTIVATION_FN, input_shape=(embedding_vector_dimension,))
            layers.Dense(128, activation='relu', input_shape=(self.embedding_vector_dimension,)),
            layers.Dense(self.image_vector_dimensions, activation=None),
            layers.Lambda(lambda x: tf.keras.backend.l2_normalize(x, axis=1)),
        ], name='siamese_projection')

        v1 = self.projection_model(emb_input_1)
        v2 = self.projection_model(emb_input_2)


        v3 = self.projection_model(emb_input_3)


        distance_AnchorPositive = layers.Lambda(cosine_distance)([v1, v2])

        distance_AnchorNegative = layers.Lambda(cosine_distance)([v1, v3])

        # computed_distance = layers.Lambda(euclidean_distance)([v1, v2])

        """ Inference model is a model from image to image vector """
        im_input = self.embedding_model.input
        embedding = self.embedding_model(im_input)
        image_vector = self.projection_model(embedding)
        self.inference_model = Model(inputs=im_input, outputs=image_vector, name='siamese_inference')

        super(SiameseOfflineTripletModel, self).__init__(
            inputs=[emb_input_1, emb_input_2, emb_input_3], outputs=[distance_AnchorPositive,distance_AnchorNegative],
            name=embedding_model.name + '_siamese_offlineTriplet_' + str(self.image_vector_dimensions)
        )


    def compile(self, optimizer=tf.keras.optimizers.RMSprop(), loss_margin=DEFAULT_MARGIN, loss=OfflineTripletLoss, **kwargs):
        super().compile(optimizer=optimizer, loss=loss(margin=loss_margin), **kwargs)

    def fit(self, x=None, y=None, epochs=NUM_EPOCHS, steps_per_epoch=STEPS_PER_EPOCH, num_classes=None, callbacks=[tensorboard_cb], **kwargs):

        if num_classes is not None and 'class_weight' not in kwargs:
            kwargs = dict(kwargs, class_weight={0: 1 / num_classes, 1: (num_classes - 1) / num_classes})

        return super().fit(x=x, y=y, epochs=epochs, steps_per_epoch=steps_per_epoch, callbacks=callbacks, **kwargs)


    @staticmethod
    def prepare_dataset(emb_vectors, emb_labels):
        stats = getLabelStats(emb_labels)

        total_labels = stats["total_labels"]
        images_per_label = stats["images_per_label"]

        tuples_per_label = int(images_per_label / 3)
        total_tuples = int(tuples_per_label * total_labels)

        emb_stats = getEmbStats(emb_vectors)
        emb_dim = emb_stats["emb_dimension"]

        # re arrange whole dataset by labels
        labels_stat = np.zeros((total_labels), dtype='uint16')
        labels_train = np.empty((total_labels, images_per_label, emb_dim), dtype='uint8')

        for i in range(len(emb_labels)):
            tr_leb = emb_labels[i]
            tr_vec = emb_vectors[i]
            labels_train[tr_leb, labels_stat[tr_leb]] = tr_vec
            labels_stat[tr_leb] += 1

        # shuffle all
        for i in range(total_labels):
            np.random.shuffle(labels_train[i])

        # create tuples
        anchor_images = np.empty((total_tuples, emb_dim), dtype='uint8')
        anchor_labels = np.empty((total_tuples), dtype='uint8')

        for i in range(total_labels):
            for j in range(tuples_per_label):
                anchor_labels[i * tuples_per_label + j] = i
                anchor_images[i * tuples_per_label + j] = labels_train[i, j]

        positive_images = np.empty((total_tuples, emb_dim), dtype='uint8')
        positive_labels = np.empty((total_tuples), dtype='uint8')

        for i in range(total_labels):
            for j in range(tuples_per_label):
                positive_labels[i * tuples_per_label + j] = i
                positive_images[i * tuples_per_label + j] = labels_train[i, tuples_per_label + j]

        negative_images = np.empty((total_tuples, emb_dim), dtype='uint8')
        negative_labels = np.empty((total_tuples), dtype='uint8')

        for i in range(total_labels):
            for j in range(tuples_per_label):
                negative_labels[i * tuples_per_label + j] = i
                negative_images[i * tuples_per_label + j] = labels_train[i, tuples_per_label * 2 + j]

        # we need to ensure we use random labels, but without images from anchor label
        shuffle_arrays([negative_labels, negative_images])

        for i in range(total_labels):
            k = ((i + 1) * tuples_per_label, 0)[i == total_labels - 1]
            for j in range(tuples_per_label):
                c = i * tuples_per_label + j
                tmp_label = negative_labels[c]

                if tmp_label == i:
                    tmp_image = negative_images[c]
                    while negative_labels[k] == i:
                        k += 1
                    negative_labels[c] = negative_labels[k]
                    negative_images[c] = negative_images[k]
                    negative_labels[k] = tmp_label
                    negative_images[k] = tmp_image

        # randomize them one more time
        for i in range(total_labels):
            shuffle_arrays([
                negative_labels[i * tuples_per_label:(i + 1) * tuples_per_label],
                negative_images[i * tuples_per_label:(i + 1) * tuples_per_label]
            ])

        dataset = (anchor_images, anchor_labels), (positive_images, positive_labels), (negative_images, negative_labels)

        embeddings_ds = tf.data.Dataset.zip((
            tf.data.Dataset.from_tensor_slices(anchor_images),
            tf.data.Dataset.from_tensor_slices(positive_images),
            tf.data.Dataset.from_tensor_slices(negative_images)
        )).cache().shuffle(1000).repeat()
        embeddings_ds = embeddings_ds.map(make_label_for_triplet, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
        return embeddings_ds.batch(TRAIN_BATCH_SIZE)