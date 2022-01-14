import sys
sys.path.append("..")

from utils.common import *
from utils.distance import *
from src.data.embeddings import *
from src.model.alexnet import AlexNetModel
from tensorflow.keras import layers, Model

alexnet = AlexNetModel()
alexnet.compile()
alexnet.load_weights(get_modeldir('alexnet_cifar10.h5'))

for layer in alexnet.layers:
    layer.trainable = False

# Filippo's Siemese model

## Model hyperparters
EMBEDDING_VECTOR_DIMENSION = 4096
IMAGE_VECTOR_DIMENSIONS = 512

emb_input_1 = layers.Input(EMBEDDING_VECTOR_DIMENSION)
emb_input_2 = layers.Input(EMBEDDING_VECTOR_DIMENSION)

# projection model is the one to use for queries (put in a sequence after the embedding-generator model above)
projection_model = tf.keras.models.Sequential([
  layers.Dense(IMAGE_VECTOR_DIMENSIONS, activation='tanh', input_shape=(EMBEDDING_VECTOR_DIMENSION,))
])

v1 = projection_model(emb_input_1)
v2 = projection_model(emb_input_2)

computed_distance = layers.Lambda(cosine_distance)([v1, v2])
# siamese is the model we train
siamese = Model(inputs=[emb_input_1, emb_input_2], outputs=computed_distance)

## Training hyperparameters (values selected randomly at the moment, would be easy to set up hyperparameter tuning wth Keras Tuner)
TRAIN_BATCH_SIZE = 128
STEPS_PER_EPOCH = 1000
NUM_EPOCHS = 3

# TODO: If there's a need to adapt the learning rate, explicitly create the optimizer instance here and pass it into compile
siamese.compile(loss=loss(margin=0.05), optimizer="RMSprop")
siamese.summary()

embeddings, embedding_labels = load_embeddings()
embeddings_ds = tf.data.Dataset.zip((
    tf.data.Dataset.from_tensor_slices(embeddings),
    tf.data.Dataset.from_tensor_slices(embedding_labels)
))
embeddings_ds = embeddings_ds.cache().shuffle(1000).repeat()

@tf.function
def make_label_for_pair(embeddings, labels):
  #embedding_1, label_1 = tuple_1
  #embedding_2, label_2 = tuple_2
  return (embeddings[0,:], embeddings[1,:]), tf.cast(labels[0] == labels[1], tf.float32)

# because of shuffling, we can take two adjacent tuples as a randomly matched pair
train_ds = embeddings_ds.window(2, drop_remainder=True)
train_ds = train_ds.flat_map(lambda w1, w2: tf.data.Dataset.zip((w1.batch(2), w2.batch(2)))) # see https://stackoverflow.com/questions/55429307/how-to-use-windows-created-by-the-dataset-window-method-in-tensorflow-2-0
# generate the target label depending on whether the labels match or not
train_ds = train_ds.map(make_label_for_pair, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
# resample to the desired distribution
# train_ds = train_ds.rejection_resample(lambda embs, target: tf.cast(target, tf.int32), [0.5, 0.5], initial_dist=[0.9, 0.1])
# train_ds = train_ds.map(lambda _, vals: vals) # discard the prepended "selected" class from the rejction resample, since we aleady have it available

embeddings_ds_size = tf.data.experimental.cardinality(embeddings_ds).numpy()
train_ds_size = tf.data.experimental.cardinality(train_ds).numpy()

ds = train_ds.batch(TRAIN_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
history = siamese.fit(ds, epochs=NUM_EPOCHS, steps_per_epoch=STEPS_PER_EPOCH)

# Build full inference model (from image to image vector):

im_input = alexnet.input
embedding = alexnet(im_input)
image_vector = projection_model(embedding)
inference_model = Model(inputs=im_input, outputs=image_vector)

inference_model.save(get_modeldir('seamese1.tf'), save_format='tf', include_optimizer=False)
