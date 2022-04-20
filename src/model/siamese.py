from src.utils.common import *
from src.utils.distance import *
from tensorflow.keras import layers, Model


class SiameseModel(Model):
    """ Filippo's Siamese model
    """

    def __init__(self, siamese_network, embedding_vector_dimension=4096, image_vector_dimensions=512):
        super().__init__()

        emb_input_1 = layers.Input(embedding_vector_dimension)
        emb_input_2 = layers.Input(embedding_vector_dimension)

        # projection model is the one to use for queries (put in a sequence after the embedding-generator model above)
        projection_model = tf.keras.models.Sequential([
            layers.Dense(image_vector_dimensions, activation='tanh', input_shape=(embedding_vector_dimension,))
        ])

        v1 = projection_model(emb_input_1)
        v2 = projection_model(emb_input_2)
        computed_distance = layers.Lambda(cosine_distance)([v1, v2])

        # siamese is the model we train
        self.siamese = Model(inputs=[emb_input_1, emb_input_2], outputs=computed_distance)
        # TODO: If there's a need to adapt the learning rate, explicitly create the optimizer instance here and pass it into compile
        self.siamese.compile(loss=loss(margin=0.05), optimizer="RMSprop")
        self.siamese.summary()

        # Build full inference model (from image to image vector):
        im_input = siamese_network.input
        embedding = siamese_network(im_input)
        image_vector = projection_model(embedding)
        self.inference_model = Model(inputs=im_input, outputs=image_vector)

    def fit(self, ds, epochs=3, steps_per_epoch=1000, **kwargs):
        return self.siamese.fit(ds, epochs=epochs, steps_per_epoch=steps_per_epoch, **kwargs)

    def save(self, filepath, overwrite=True, include_optimizer=False, save_format=None,
             signatures=None, options=None, save_traces=True):
        return self.inference_model.save(filepath, overwrite=overwrite, include_optimizer=include_optimizer,
                                         save_format=save_format, signatures=signatures, options=options,
                                         save_traces=save_traces)

    def call(self, inputs, training=None, mask=None):
        return self.inference_model(inputs)