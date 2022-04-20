import sys

sys.path.append("..")

from src.data.embeddings import *
from utils.common import *
from utils.distance import *
from src.model.alexnet import AlexNetModel
from src.model.siamese import SiameseModel
from tensorflow.keras import layers, Model

model_name = 'imagenette_alexnet'
embeddings_name = model_name + '_embeddings'

from pathlib import Path
import tarfile
import numpy as np

batch_size = 32
img_height = 426
img_width = 320

data_dir = Path('../datasets/imagenette2-320/train')
print(data_dir.absolute())
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir, image_size=(227, 227), batch_size=batch_size)

# load model
# alexnet = models.load_model(get_modeldir(model_name + '.tf'))

# create model
# alexnet = AlexNetModel()
# alexnet.compile()
# alexnet.summary()

# load weights
# alexnet.load_weights(get_modeldir(model_name + '.h5'))

# train
# alexnet.fit(train_ds, validation_data=test_ds)

# save
# alexnet.save_weights(get_modeldir(model_name + '.h5'))
# alexnet.save(get_modeldir(model_name + '.tf'))

# evaluate
# alexnet.evaluate(validation_ds)
# res = alexnet.predict(validation_ds)

# for layer in alexnet.layers:
#     layer.trainable = False

# embeddings, embedding_labels = calc_embeddings(alexnet)
# save_embeddings(embeddings, embedding_labels, embeddings_name)

print("Done.")
