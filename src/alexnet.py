import numpy as np
from keras import Model
import tensorflow as tf
from tensorflow.keras import datasets
from src.model.alexnet import AlexNetModel
from src.utils.common import get_modeldir, process_images_couple, get_datadir

model_name = 'alexnet_cifar10-new'
train_ds, test_ds, validation_ds = AlexNetModel.x_dataset()

# load model
# alexnet = models.load_model(get_modeldir(model_name + '.tf'))

# create model
alexnet = AlexNetModel()
alexnet.compile()
# alexnet.summary()

# load weights
alexnet.load_weights(get_modeldir(model_name + '.h5'))

# train
# alexnet.fit(train_ds, validation_data=test_ds)

# save
# alexnet.save_weights(get_modeldir(model_name + '.h5'))
# alexnet.save(get_modeldir(model_name + '.tf'))

# evaluate
alexnet.evaluate(validation_ds)
# res = alexnet.predict(validation_ds)

print('done')
