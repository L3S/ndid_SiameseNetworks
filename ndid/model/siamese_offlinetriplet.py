from ndid.utils.common import *
from ndid.utils.tuple import produce_tuples
from ndid.utils.distance import cosine_distance, euclidean_distance
from tensorflow.keras import layers, callbacks, Model
from ndid.loss.offline_triplet import OfflineTripletLoss

tensorboard_cb = callbacks.TensorBoard(get_logdir('siamese/fit'), histogram_freq=1)

EMBEDDING_VECTOR_DIMENSION = 4096
IMAGE_VECTOR_DIMENSIONS = 3  # use for test visualization on tensorboard
ACTIVATION_FN = 'tanh'  # same as in paper
DEFAULT_MARGIN = 0.5

NUM_EPOCHS = 10
TRAIN_BATCH_SIZE = 64
STEPS_PER_EPOCH = 100

class SiameseOfflineTripletModel(Model):
    def __init__(self, embedding_model, image_vector_dimensions=IMAGE_VECTOR_DIMENSIONS, loss_margin=DEFAULT_MARGIN, fit_epochs=NUM_EPOCHS, basename=None):
        if basename is None:
            basename = embedding_model.name + '_d' + str(image_vector_dimensions) + '_m' + str(loss_margin) + '_s' + str(fit_epochs * STEPS_PER_EPOCH)

        embedding_vector_dimension = embedding_model.output_shape[1]
        emb_input_1 = layers.Input(embedding_vector_dimension)
        emb_input_2 = layers.Input(embedding_vector_dimension)
        emb_input_3 = layers.Input(embedding_vector_dimension)

        """ projection model is the one to use for queries (put in a sequence after the embedding-generator model) """
        projection_model = tf.keras.models.Sequential([
            # layers.Dense(image_vector_dimensions, activation=ACTIVATION_FN, input_shape=(embedding_vector_dimension,))
            layers.Dense(128, activation='relu', input_shape=(embedding_vector_dimension,)),
            layers.Dense(image_vector_dimensions, activation=None),
            # TODO: remove normalization when play with distance formula
            # layers.Lambda(lambda x: tf.keras.backend.l2_normalize(x, axis=1)),
        ], name='siamese_projection_' + basename)

        v1 = projection_model(emb_input_1)
        v2 = projection_model(emb_input_2)
        v3 = projection_model(emb_input_3)

        distance_pos = layers.Lambda(cosine_distance)([v1, v2])
        distance_neg = layers.Lambda(cosine_distance)([v1, v3])

        # distance_pos = layers.Lambda(euclidean_distance)([v1, v2])
        # distance_neg = layers.Lambda(euclidean_distance)([v1, v3])

        """ Inference model is a model from image to image vector """
        im_input = embedding_model.input
        embedding = embedding_model(im_input)
        image_vector = projection_model(embedding)
        inference_model = Model(inputs=im_input, outputs=image_vector, name='siamese_inference_' + basename)

        super(SiameseOfflineTripletModel, self).__init__(
            inputs=[emb_input_1, emb_input_2, emb_input_3], outputs=[distance_pos, distance_neg],
            name='siamese_' + basename
        )

        self.loss_margin = loss_margin
        self.fit_epochs = fit_epochs
        self.projection_model = projection_model
        self.inference_model = inference_model

    def compile(self, loss=OfflineTripletLoss, loss_margin=None, optimizer=tf.keras.optimizers.RMSprop(), **kwargs):

        if loss_margin is None:
            loss_margin = self.loss_margin

        super().compile(optimizer=optimizer, loss=loss(margin=loss_margin), **kwargs)

    def fit(self, x=None, y=None, epochs=None, steps_per_epoch=STEPS_PER_EPOCH, num_classes=None, callbacks=[tensorboard_cb], **kwargs):

        if epochs is None:
            epochs = self.fit_epochs

        if num_classes is not None and 'class_weight' not in kwargs:
            kwargs = dict(kwargs, class_weight={0: 1 / num_classes, 1: (num_classes - 1) / num_classes})

        return super().fit(x=x, y=y, epochs=epochs, steps_per_epoch=steps_per_epoch, callbacks=callbacks, **kwargs)

    @staticmethod
    def prepare_dataset(emb_vectors, emb_labels, batch_size=TRAIN_BATCH_SIZE):
        (anchor_images, anchor_labels), (positive_images, positive_labels), (negative_images, negative_labels) = produce_tuples(emb_vectors, emb_labels)

        embeddings_ds = tf.data.Dataset.zip((
            tf.data.Dataset.from_tensor_slices(anchor_images),
            tf.data.Dataset.from_tensor_slices(positive_images),
            tf.data.Dataset.from_tensor_slices(negative_images)
        )).cache().shuffle(1000).repeat()

        @tf.function
        def make_label_for_triplet(emb1, emb2, emb3):
            return ((emb1, emb2, emb3)), tf.cast(0, tf.float32)

        embeddings_ds = embeddings_ds.map(make_label_for_triplet, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
        return embeddings_ds.batch(batch_size)