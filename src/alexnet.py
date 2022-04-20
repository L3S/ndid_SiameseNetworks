import sys
sys.path.append("..")

from src.data.embeddings import *
from utils.common import *
from utils.distance import *
from src.model.alexnet import AlexNetModel
from src.model.siamese import SiameseModel
from tensorflow.keras import layers, Model

model_name = 'cifar10_alexnet'
embeddings_name = model_name + '_embeddings'
train_ds, test_ds, validation_ds = AlexNetModel.x_dataset()

# load model
# alexnet = models.load_model(get_modeldir(model_name + '.tf'))

# create model
alexnet = AlexNetModel()
alexnet.compile()
alexnet.summary()

exit()
# load weights
alexnet.load_weights(get_modeldir(model_name + '.h5'))

# train
# alexnet.fit(train_ds, validation_data=test_ds)

# save
# alexnet.save_weights(get_modeldir(model_name + '.h5'))
# alexnet.save(get_modeldir(model_name + '.tf'))

# evaluate
# alexnet.evaluate(validation_ds)
# res = alexnet.predict(validation_ds)

for layer in alexnet.layers:
    layer.trainable = False

# embeddings, embedding_labels = calc_embeddings(alexnet)
# save_embeddings(embeddings, embedding_labels, embeddings_name)


# Model hyperparters
EMBEDDING_VECTOR_DIMENSION = 4096
IMAGE_VECTOR_DIMENSIONS = 512


alexnet = Model(inputs=alexnet.input, outputs=alexnet.layers[-3].output)


## Training hyperparameters (values selected randomly at the moment, would be easy to set up hyperparameter tuning wth Keras Tuner)
TRAIN_BATCH_SIZE = 128
STEPS_PER_EPOCH = 1000
NUM_EPOCHS = 25

embeddings, embedding_labels = load_embeddings(embeddings_name)
embeddings_ds = tf.data.Dataset.zip((
    tf.data.Dataset.from_tensor_slices(embeddings),
    tf.data.Dataset.from_tensor_slices(embedding_labels)
))
embeddings_ds = embeddings_ds.cache().shuffle(1000).repeat()

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

ds = train_ds.batch(TRAIN_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
siamese = SiameseModel(alexnet, EMBEDDING_VECTOR_DIMENSION, IMAGE_VECTOR_DIMENSIONS)
history = siamese.fit(ds, epochs=NUM_EPOCHS, steps_per_epoch=STEPS_PER_EPOCH)

siamese.save(get_modeldir('cifar10_alexnet_seamese25_' + str(IMAGE_VECTOR_DIMENSIONS) + '.tf'), save_format='tf')

embedding_vds = cifar10_complete_resized().batch(batch_size=32, drop_remainder=False)
embeddings = siamese.predict(embedding_vds)
labels = np.concatenate([y for x, y in embedding_vds], axis=0)
save_embeddings(embeddings, labels, 'cifar10_alexnet_embeddings_siamese25')

print('done')
