import sys
sys.path.append("..")

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from src.data.simple3 import load_dataset3, NUM_CLASSES
from src.utils.embeddings import project_embeddings, calc_vectors, save_embeddings
from src.utils.common import get_modeldir
from src.model.alexnet import AlexNetModel, TARGET_SHAPE
from src.model.siamese import SiameseModel

model_name = 'simple3_alexnet'
embeddings_name = model_name + '_embeddings'

train_ds, val_ds, test_ds = load_dataset3(image_size=TARGET_SHAPE, preprocess_fn=AlexNetModel.preprocess_input)
comb_ds = train_ds.concatenate(val_ds).concatenate(test_ds)

# create model
model = AlexNetModel(NUM_CLASSES)
model.compile()
model.summary()

# load weights
model.load_weights(get_modeldir(model_name + '.h5'))

# train & save model
# model.fit(train_ds, validation_data=val_ds)
# model.save_weights(get_modeldir(model_name + '.h5'))

# evaluate
# print('evaluating...')
# model.evaluate(test_ds)

for layer in model.layers:
    layer.trainable = False

print('calculating embeddings...')
embedding_model = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)
embedding_model.summary()

emb_vectors, emb_labels = calc_vectors(comb_ds, embedding_model)
# project_embeddings(emb_vectors, emb_labels, model_name + '_emb')
save_embeddings(emb_vectors, emb_labels, embeddings_name)

# emb_vectors, emb_labels = load_embeddings(embeddings_name)

# siamese is the model we train
siamese = SiameseModel(embedding_vector_dimension=4096, image_vector_dimensions=3)
siamese.compile(loss_margin=0.1)  # TODO: experiment with high value, e.g. 2
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

dstm = distance_matrix(siamese_vectors, siamese_vectors)
print('Under the margin', (dstm < 0.1).sum() / 2)


fig = plt.figure()
ax = fig.add_subplot(projection='3d')

_pts = np.random.uniform(size=[500, 3], low=-1, high=1)
_pts = _pts / np.linalg.norm(_pts, axis=-1)[:, None]
ax.scatter(_pts[:, 0], _pts[:, 1], _pts[:, 2])

ax.scatter(siamese_vectors[:, 0], siamese_vectors[:, 1], siamese_vectors[:, 2])
ax.set_xlim([-1,1])
ax.set_ylim([-1,1])
ax.set_zlim([-1,1])
ax.invert_zaxis()

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()