import time
import os
import tensorflow as tf
from sidd.data.imagenet1k import ImageNetDataLoader
from sidd.model.vit_custom import VitModel
import tensorflow_datasets as tfds
from sidd.utils.common import get_dataset

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

# Create AlexNet Model (Fused)
model = VitModel(input_shape=(224, 224), num_classes=1000)
model.summary()

# Constants
BATCH_SIZE = 512

# Init Data Loaders
train_data_loader = ImageNetDataLoader(split=tfds.Split.TRAIN)
val_data_loader = ImageNetDataLoader(split=tfds.Split.VALIDATION)
test_data_loader = ImageNetDataLoader(split=tfds.Split.TEST)

train_generator = train_data_loader.dataset_generator(batch_size=BATCH_SIZE, augment=False)
val_generator = val_data_loader.dataset_generator(batch_size=BATCH_SIZE, augment=False)
test_generator = test_data_loader.dataset_generator(batch_size=BATCH_SIZE, augment=False)

model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=[
        tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
        tf.keras.metrics.TopKCategoricalAccuracy(5, name="top-5-accuracy"),
    ]
)

start = time.time()
history = model.fit(
    x=train_generator,
    steps_per_epoch=train_data_loader.get_num_steps(),
    validation_data=val_generator,
    validation_steps=val_data_loader.get_num_steps(),
)
print('Model %s trained in %ss' % (model.name, time.time() - start))
model.save_weights('./models/weights/vit_imagenet2')

_, accuracy, top_5_accuracy = model.evaluate(test_generator, steps=test_data_loader.get_num_steps())
print(f"Test accuracy: {round(accuracy * 100, 2)}%")
print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

