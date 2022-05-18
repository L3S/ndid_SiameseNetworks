import sys
sys.path.append("..")

import numpy as np
import tensorflow as tf
from src.data.imagenette import load_dataset3, NUM_CLASSES
from src.utils.embeddings import save_embeddings, project_embeddings
from src.utils.common import get_modeldir
from src.model.vgg16 import VGG16Model, PRETRAIN_EPOCHS
from src.model.siamese import SiameseModel

model_name = 'imagenette_vgg16_small'
embeddings_name = model_name + '_embeddings'

TARGET_SHAPE = (32, 32)

train_ds, val_ds, test_ds = load_dataset3(image_size=TARGET_SHAPE, preprocess_fn=VGG16Model.preprocess_input)
PRETRAIN_TOTAL_STEPS = PRETRAIN_EPOCHS * len(train_ds)

# create model
model = tf.keras.applications.VGG16(
    include_top=True,
    input_shape=(32, 32, 3),
    weights=None,
    classes=10
)

PRETRAIN_EPOCHS = 20
PRETRAIN_TOTAL_STEPS = PRETRAIN_EPOCHS * len(train_ds)

model.compile(optimizer=tf.keras.optimizers.RMSprop(tf.keras.optimizers.schedules.CosineDecay(1e-3, PRETRAIN_TOTAL_STEPS)))
model.summary()

# load weights
# model.load_weights(get_modeldir(model_name + '.h5'))

# train & save model
model.fit(train_ds, epochs=PRETRAIN_EPOCHS, validation_data=val_ds)
model.save_weights(get_modeldir(model_name + '.h5'))

# evaluate
print('evaluating...')
model.evaluate(test_ds)

for layer in model.layers:
    layer.trainable = False

# save embeddings
embedding_model = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)
embedding_model.summary()

embedding_vds = train_ds.concatenate(val_ds).concatenate(test_ds)
print('calculating embeddings...')
embeddings = embedding_model.predict(embedding_vds)
embedding_labels = np.concatenate([y for x, y in embedding_vds], axis=0)
save_embeddings(embeddings, embedding_labels, embeddings_name)

# embeddings, embedding_labels = load_embeddings(embeddings_name)

# siamese is the model we train
siamese = SiameseModel(embedding_vector_dimension=4096, image_vector_dimensions=3)
siamese.compile(loss_margin=0.05)
siamese.summary()

## Training hyperparameters (values selected randomly at the moment, would be easy to set up hyperparameter tuning wth Keras Tuner)
## We have 128 pairs for each epoch, thus in total we will have 128 x 2 x 1000 images to give to the siamese
NUM_EPOCHS = 3
TRAIN_BATCH_SIZE = 128
STEPS_PER_EPOCH = 1000

ds = SiameseModel.prepare_dataset(embeddings, embedding_labels)
history = siamese.fit(
    ds,
    epochs=NUM_EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    class_weight={0: 1 / NUM_CLASSES, 1: (NUM_CLASSES - 1) / NUM_CLASSES}
)

# Build full inference model (from image to image vector):
inference_model = siamese.get_inference_model(embedding_model)
inference_model.save(get_modeldir(model_name + '_inference.tf'), save_format='tf', include_optimizer=False)

# inference_model = tf.keras.models.load_model(get_modeldir(model_name + '_inference.tf'), compile=False)


print('visualization')
# compute vectors of the images and their labels, store them in a tsv file for visualization
image_vectors = inference_model.predict(embedding_vds)
project_embeddings(image_vectors, embedding_labels, model_name)
