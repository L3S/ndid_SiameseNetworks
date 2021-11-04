from common import *
from cifar10_tuples import *
from alexnet import AlexNet
from tensorflow.keras import datasets, layers, models, losses, callbacks, applications, optimizers, metrics, Model
from tensorflow.keras.applications import resnet

run_suffix = '-04'

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

validation_images, validation_labels = train_images[:5000], train_labels[:5000]
train_images, train_labels = train_images[5000:], train_labels[5000:]

alexnet_train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
alexnet_test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
alexnet_validation_ds = tf.data.Dataset.from_tensor_slices((validation_images, validation_labels))

train_ds_size = tf.data.experimental.cardinality(alexnet_train_ds).numpy()
test_ds_size = tf.data.experimental.cardinality(alexnet_test_ds).numpy()
validation_ds_size = tf.data.experimental.cardinality(alexnet_validation_ds).numpy()
print("Training data size:", train_ds_size)
print("Test data size:", test_ds_size)
print("Validation data size:", validation_ds_size)

alexnet_train_ds = (alexnet_train_ds.map(process_images_couple).shuffle(buffer_size=train_ds_size).batch(batch_size=32, drop_remainder=True))
alexnet_test_ds = (alexnet_test_ds.map(process_images_couple).shuffle(buffer_size=train_ds_size).batch(batch_size=32, drop_remainder=True))
alexnet_validation_ds = (alexnet_validation_ds.map(process_images_couple).shuffle(buffer_size=train_ds_size).batch(batch_size=32, drop_remainder=True))

# plot_first5_fig(alexnet_train_ds)
# plot_first5_fig(alexnet_test_ds)
# plot_first5_fig(alexnet_validation_ds)

alexnet = AlexNet()
# alexnet.train_model(alexnet_train_ds, alexnet_validation_ds, alexnet_test_ds)
# alexnet.save_model('alexnet_cifar10' + run_suffix)
alexnet.load_model('alexnet_cifar10' + run_suffix)

print('alexnet evaluate')
alexnet.get_model().evaluate(alexnet_test_ds)

# (anchor_images, anchor_labels), (positive_images, positive_labels), (negative_images, negative_labels) = produce_tuples()
# save_tuples(anchor_images, anchor_labels, positive_images, positive_labels, negative_images, negative_labels)

# (anchor_images, anchor_labels), (positive_images, positive_labels), (negative_images, negative_labels) = load_tuples()
tuples_ds = prepare_dataset()
tuples_ds_size = tf.data.experimental.cardinality(tuples_ds).numpy()

# sample = next(iter(tuples_ds))
# visualize(*sample)

# Let's now split our dataset in train and validation.
siamese_train_ds = tuples_ds.take(round(tuples_ds_size * 0.8))
siamese_validation_ds = tuples_ds.skip(round(tuples_ds_size * 0.8))

class DistanceLayer(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):
        ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
        an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
        return (ap_distance, an_distance)


anchor_input = layers.Input(name="anchor", shape=target_shape + (3,))
positive_input = layers.Input(name="positive", shape=target_shape + (3,))
negative_input = layers.Input(name="negative", shape=target_shape + (3,))

alexnet_model = alexnet.get_model()
distances = DistanceLayer()(
    alexnet_model(resnet.preprocess_input(anchor_input)),
    alexnet_model(resnet.preprocess_input(positive_input)),
    alexnet_model(resnet.preprocess_input(negative_input)),
)

siamese_network = Model(
    inputs=[anchor_input, positive_input, negative_input], outputs=distances
)

"""
## Putting everything together

We now need to implement a model with custom training loop so we can compute
the triplet loss using the three embeddings produced by the Siamese network.

Let's create a `Mean` metric instance to track the loss of the training process.
"""


class SiameseModel(Model):
    """The Siamese Network model with a custom training and testing loops.

    Computes the triplet loss using the three embeddings produced by the Siamese Network.

    The triplet loss is defined as:
       L(A, P, N) = max(‖f(A) - f(P)‖² - ‖f(A) - f(N)‖² + margin, 0)
    """

    def __init__(self, siamese_network, margin=0.5):
        super(SiameseModel, self).__init__()
        self.siamese_network = siamese_network
        self.margin = margin
        self.loss_tracker = metrics.Mean(name="loss")

    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        # GradientTape is a context manager that records every operation that
        # you do inside. We are using it here to compute the loss so we can get
        # the gradients and apply them using the optimizer specified in
        # `compile()`.
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)

        # Storing the gradients of the loss function with respect to the
        # weights/parameters.
        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)

        # Applying the gradients on the model using the specified optimizer
        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_weights)
        )

        # Let's update and return the training loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self._compute_loss(data)

        # Let's update and return the loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def _compute_loss(self, data):
        # The output of the network is a tuple containing the distances
        # between the anchor and the positive example, and the anchor and
        # the negative example.
        ap_distance, an_distance = self.siamese_network(data)

        # Computing the Triplet Loss by subtracting both distances and
        # making sure we don't get a negative value.
        loss = ap_distance - an_distance
        loss = tf.maximum(loss + self.margin, 0.0)
        return loss

    @property
    def metrics(self):
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [self.loss_tracker]


"""
## Training

We are now ready to train our model.
"""

tensorboard_cb = callbacks.TensorBoard(get_logdir("siamese/fit"))

siamese_model = SiameseModel(siamese_network)
siamese_model.compile(optimizer=optimizers.Adam(0.0001))
siamese_model.fit(siamese_train_ds, epochs=10, validation_data=siamese_validation_ds, callbacks=[tensorboard_cb])

# print('saving siamese')
# siamese_model.save(get_modeldir('siamese_cifar10' + run_suffix))
# ValueError: Model <__main__.SiameseModel object at 0x7f9070531730> cannot be saved because the input shapes have not been set. Usually, input shapes are automatically determined from calling `.fit()` or `.predict()`. To manually set the shapes, call `model.build(input_shape)`.

print('saving alexnet2')
alexnet_model.save(get_modeldir('alexnet2_cifar10' + run_suffix))

# print('siamese evaluate')
# siamese_model.evaluate(alexnet_test_ds)
# ValueError: Layer model expects 3 input(s), but it received 2 input tensors. Inputs received: [<tf.Tensor 'IteratorGetNext:0' shape=(32, 227, 227, 3) dtype=float32>, <tf.Tensor 'IteratorGetNext:1' shape=(32, 1) dtype=uint8>]

print('alexnet evaluate')
alexnet_model.evaluate(alexnet_test_ds)

"""
## Inspecting what the network has learned

At this point, we can check how the network learned to separate the embeddings
depending on whether they belong to similar images.

We can use [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity) to measure the
similarity between embeddings.

Let's pick a sample from the dataset to check the similarity between the
embeddings generated for each image.
"""
sample = next(iter(siamese_train_ds))
# visualize(*sample)

anchor, positive, negative = sample
anchor_embedding, positive_embedding, negative_embedding = (
    alexnet_model(resnet.preprocess_input(anchor)),
    alexnet_model(resnet.preprocess_input(positive)),
    alexnet_model(resnet.preprocess_input(negative)),
)

"""
Finally, we can compute the cosine similarity between the anchor and positive
images and compare it with the similarity between the anchor and the negative
images.

We should expect the similarity between the anchor and positive images to be
larger than the similarity between the anchor and the negative images.
"""

cosine_similarity = metrics.CosineSimilarity()

positive_similarity = cosine_similarity(anchor_embedding, positive_embedding)
print("Positive similarity:", positive_similarity.numpy())

negative_similarity = cosine_similarity(anchor_embedding, negative_embedding)
print("Negative similarity", negative_similarity.numpy())
