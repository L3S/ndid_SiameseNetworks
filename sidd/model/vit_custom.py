import tensorflow as tf
from tensorflow.keras import layers, callbacks, Sequential, Model

from sidd.model import AsbModel
from sidd.utils.common import get_logdir, get_modeldir

tensorboard_cb = callbacks.TensorBoard(get_logdir("vit/fit"))

# DATA
BATCH_SIZE = 512
TARGET_SHAPE = (224, 224)

# TRAINING
PRETRAIN_EPOCHS = 100
EMBEDDING_VECTOR_DIMENSION = 1000

# AUGMENTATION
IMAGE_SIZE = 72  # We'll resize input images to this size
PATCH_SIZE = 6  # Size of the patches to be extract from the input images
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2

# OPTIMIZER
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001

# ARCHITECTURE
TRANSFORMER_LAYERS = 8
PROJECTION_DIM = 64
NUM_HEADS = 4
TRANSFORMER_UNITS = [
    PROJECTION_DIM * 2,
    PROJECTION_DIM,
]  # Size of the transformer layers
MLP_HEAD_UNITS = [2048, 1024]  # Size of the dense layers of the final classifier


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        BATCH_SIZE = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [BATCH_SIZE, -1, patch_dims])
        return patches


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


def create_vit_classifier(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    # Augment data.
    augmented = Sequential([
        layers.Normalization(),
        layers.Resizing(IMAGE_SIZE, IMAGE_SIZE),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(factor=0.02),
        layers.RandomZoom(
            height_factor=0.2, width_factor=0.2
        ),
    ], name="data_augmentation")(inputs)
    # Create patches.
    patches = Patches(PATCH_SIZE)(augmented)
    # Encode patches.
    encoded_patches = PatchEncoder(NUM_PATCHES, PROJECTION_DIM)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(TRANSFORMER_LAYERS):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=NUM_HEADS, key_dim=PROJECTION_DIM, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=TRANSFORMER_UNITS, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=MLP_HEAD_UNITS, dropout_rate=0.5)
    # Classify outputs.
    logits = layers.Dense(num_classes)(features)
    # Create the Keras model.
    model = Model(inputs=inputs, outputs=logits)
    return model


class VitModel(Model):
    def __init__(self, input_shape=TARGET_SHAPE, num_classes=10, weights="imagenet", train_size=None, **kwargs):
        model = create_vit_classifier(input_shape=input_shape + (3,), num_classes=num_classes)

        super(VitModel, self).__init__(inputs=model.input, outputs=model.output, name='vit')

    def compile(self,
            optimizer=tf.keras.optimizers.experimental.AdamW(
                learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY
            ),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[
                tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                tf.keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
            ],
        **kwargs):

        super().compile(optimizer=optimizer, loss=loss, metrics=metrics, **kwargs)

    def fit(self, epochs=PRETRAIN_EPOCHS, callbacks=[tensorboard_cb], **kwargs):
        return super().fit(epochs=epochs, callbacks=callbacks, **kwargs)

    def get_embedding_model(self):
        return self

    @staticmethod
    def get_target_shape():
        return TARGET_SHAPE

    @staticmethod
    def preprocess_input(image, label):
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, TARGET_SHAPE, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        image = (image - 0.5) * 2  # ViT requires images in range [-1,1]
        return image, label
