from src.utils.common import *
import tensorflow_addons as tfa
from src.utils.distance import cosine_distance, euclidean_distance
from tensorflow.keras import layers, callbacks, Model

tensorboard_cb = callbacks.TensorBoard(get_logdir('siamese/fit'))

EMBEDDING_VECTOR_DIMENSION = 4096
IMAGE_VECTOR_DIMENSIONS = 3  # use for test visualization on tensorboard
ACTIVATION_FN = 'tanh'  # same as in paper
DEFAULT_MARGIN = 0.5

NUM_EPOCHS = 3
TRAIN_BATCH_SIZE = 128
STEPS_PER_EPOCH = 100


class SiameseModel(Model):
    """ Filippo's Siamese model

    The `embedding_model` is a model used to extract embeddings, also used to create `inference_model`.
    The `projection_model` is the part of the network that generates the final image vector, uses embeddings as input.
    The `inference_model` is combined model, uses `embedding_model`'s input and `projection_model`'s output.
    The `siamese` model is the one we train. It applies the projection model to two embeddings,
    calculates the euclidean distance between the two generated image vectors and calculates the contrastive loss.
    """

    def __init__(self, embedding_model, image_vector_dimensions=IMAGE_VECTOR_DIMENSIONS, loss_margin=DEFAULT_MARGIN, fit_epochs=NUM_EPOCHS):
        embedding_vector_dimension = embedding_model.output_shape[1]
        emb_input_1 = layers.Input(embedding_vector_dimension)
        emb_input_2 = layers.Input(embedding_vector_dimension)

        # projection model is the one to use for queries (put in a sequence after the embedding-generator model above)
        projection_model = tf.keras.models.Sequential([
            # layers.Dense(image_vector_dimensions, activation=ACTIVATION_FN, input_shape=(embedding_vector_dimension,))
            layers.Dense(128, activation='relu', input_shape=(embedding_vector_dimension,)),
            layers.Dense(image_vector_dimensions, activation=None),
            layers.Lambda(lambda x: tf.keras.backend.l2_normalize(x, axis=1)),
        ], name='siamese_projection')

        v1 = projection_model(emb_input_1)
        v2 = projection_model(emb_input_2)

        # As a note, [here](https://towardsdatascience.com/contrastive-loss-explaned-159f2d4a87ec) they mention that
        # cosine distance is preferable to euclidean distance: in a large dimensional space, all points tend to be far
        # apart by the euclidian measure. In higher dimensions, the angle between vectors is a more effective measure.
        # Note that, when using cosine distance, the margin needs to be reduced from its default value of 1 (see below)
        computed_distance = layers.Lambda(cosine_distance)([v1, v2])
        # computed_distance = layers.Lambda(euclidean_distance)([v1, v2])

        """ Inference model is a model from image to image vector """
        im_input = embedding_model.input
        embedding = embedding_model(im_input)
        image_vector = projection_model(embedding)
        inference_model = Model(inputs=im_input, outputs=image_vector, name='siamese_inference')

        super(SiameseModel, self).__init__(
            inputs=[emb_input_1, emb_input_2], outputs=computed_distance,
            name=embedding_model.name + '_siamese_d' + str(image_vector_dimensions) + '_m' + str(loss_margin) + '_s' + str(fit_epochs * STEPS_PER_EPOCH)
        )

        self.loss_margin = loss_margin
        self.fit_epochs = fit_epochs
        self.projection_model = projection_model
        self.inference_model = inference_model

    def compile(self, optimizer=tf.keras.optimizers.RMSprop(), loss_margin=None, **kwargs):

        if loss_margin is None:
            loss_margin = self.loss_margin

        super().compile(optimizer=optimizer, loss=tfa.losses.ContrastiveLoss(margin=loss_margin), **kwargs)

    def fit(self, x=None, y=None, epochs=None, steps_per_epoch=STEPS_PER_EPOCH, num_classes=None, callbacks=[tensorboard_cb], **kwargs):

        if epochs is None:
            epochs = self.fit_epochs

        if num_classes is not None and 'class_weight' not in kwargs:
            kwargs = dict(kwargs, class_weight={0: 1 / num_classes, 1: (num_classes - 1) / num_classes})

        return super().fit(x=x, y=y, epochs=epochs, steps_per_epoch=steps_per_epoch, callbacks=callbacks, **kwargs)

    @staticmethod
    def prepare_dataset(emb_vectors, emb_labels):
        """
        To train the siamese networks, we need to generate random pairs of embeddings,
        assigning as target `1` if the two come from the same class and `0` otherwise.
        """

        # zip together embeddings and labels, cache in memory (maybe not necessary), shuffle, repeat forever
        embeddings_ds = tf.data.Dataset.zip((
            tf.data.Dataset.from_tensor_slices(emb_vectors),
            tf.data.Dataset.from_tensor_slices(emb_labels)
        )).cache().shuffle(1000).repeat()

        @tf.function
        def make_label_for_pair(embeddings, labels):
            return (embeddings[0, :], embeddings[1, :]), tf.cast(labels[0] == labels[1], tf.uint8)

        # because of shuffling, we can take two adjacent tuples as a randomly matched pair
        # each "window" is a dataset that contains a subset of elements of the input dataset
        windows_ds = embeddings_ds.window(2, drop_remainder=True)
        # https://stackoverflow.com/questions/55429307/how-to-use-windows-created-by-the-dataset-window-method-in-tensorflow-2-0
        flat_ds = windows_ds.flat_map(lambda w1, w2: tf.data.Dataset.zip((w1.batch(2), w2.batch(2))))
        # generate the target label depending on whether the labels match or not
        map_ds = flat_ds.map(make_label_for_pair, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
        train_ds = map_ds.batch(TRAIN_BATCH_SIZE)  # .prefetch(tf.data.AUTOTUNE)
        return train_ds
