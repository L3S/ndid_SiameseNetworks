import sys
sys.path.append("..")

import tensorflow as tf
from src.data.cifar10 import load_dataset3, NUM_CLASSES
from src.utils.embeddings import save_embeddings, project_embeddings, calc_vectors
from src.utils.common import get_modeldir
from src.model.vgg16 import VGG16Model, PRETRAIN_EPOCHS, EMBEDDING_VECTOR_DIMENSION
from src.model.siamese import SiameseModel

model_name = 'cifar10_vgg16'
embeddings_name = model_name + '_embeddings'

TARGET_SHAPE = (32, 32)

train_ds, val_ds, test_ds = load_dataset3(image_size=TARGET_SHAPE, preprocess_fn=VGG16Model.preprocess_input)
comb_ds = train_ds.concatenate(val_ds).concatenate(test_ds)
PRETRAIN_TOTAL_STEPS = PRETRAIN_EPOCHS * len(train_ds)

# create model
model = VGG16Model(input_shape=TARGET_SHAPE)
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

print('calculating embeddings...')
embedding_model = model.get_embedding_model()
embedding_model.summary()

emb_vectors, emb_labels = calc_vectors(comb_ds, embedding_model)
save_embeddings(emb_vectors, emb_labels, embeddings_name)

# emb_vectors, emb_labels = load_embeddings(embeddings_name)

# siamese is the model we train
siamese = SiameseModel(embedding_vector_dimension=EMBEDDING_VECTOR_DIMENSION, image_vector_dimensions=3)
siamese.compile(loss_margin=0.05)
siamese.summary()

ds = SiameseModel.prepare_dataset(emb_vectors, emb_labels)
history = siamese.fit(ds, class_weight={0: 1 / NUM_CLASSES, 1: (NUM_CLASSES - 1) / NUM_CLASSES})

# Build full inference model (from image to image vector):
inference_model = siamese.get_inference_model(embedding_model)
inference_model.save(get_modeldir(model_name + '_inference.tf'), save_format='tf', include_optimizer=False)

# inference_model = tf.keras.models.load_model(get_modeldir(model_name + '_inference.tf'), compile=False)


print('visualization')
# compute vectors of the images and their labels, store them in a tsv file for visualization
siamese_vectors, siamese_labels = calc_vectors(comb_ds, inference_model)
project_embeddings(siamese_vectors, siamese_labels, model_name + '_siamese')
