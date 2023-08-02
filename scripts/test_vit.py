import time
import os
import tensorflow as tf
from sidd.model.vit_custom import VitModel
from tensorflow.keras import layers, datasets, callbacks, Sequential, Model
import tensorflow_datasets as tfds
from sidd.utils.common import get_dataset

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

(x_train, y_train), (x_test, y_test) = datasets.cifar100.load_data()

print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")

# Create AlexNet Model (Fused)
model = VitModel(input_shape=(32, 32), num_classes=100)
model.summary()

# Compile & Train
model.compile()

start = time.time()
history = model.fit(x=x_train, y=y_train)
print('Model %s trained in %ss' % (model.name, time.time() - start))
model.save_weights('./models/weights/vit_cifar10')

_, accuracy, top_5_accuracy = model.evaluate(x_test, y_test)
print(f"Test accuracy: {round(accuracy * 100, 2)}%")
print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")
