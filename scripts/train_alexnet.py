import time
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import tensorflow as tf
from sidd.model.alexnet import create_alexnet_model
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
from sidd.utils.common import get_dataset

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
plt.rcParams["figure.figsize"] = 30, 30


class ImageNetDataLoader:
    def __init__(
            self,
            source_data_dir: str = str(get_dataset('imagenet')),
            split: str = tfds.Split.TRAIN,
            image_dims: tuple = (224, 224),
            num_classes=1000
    ) -> None:
        """
        __init__
        - Instance Variable Initialization
        - Download and Set Up Dataset (One Time Operation)
        - Use TFDS to Load and convert the ImageNet Dataset

        Args:
            source_data_dir (str): Path to Downloaded tar files
            spliit (str): Split to load as. Eg. train, test, train[:80%]. Defaults to "train"
            image_dims (tuple, optional): Image Dimensions (width & height). Defaults to (224, 224).
            num_classes (int): Number of Classes contained in this dataset. Defaults to 1000
        """

        # Constants
        self.NUM_CLASSES = num_classes
        self.BATCH_SIZE = None
        self.NUM_CHANNELS = 3
        self.LABELS = []
        self.LABELMAP = {}
        self.AUTOTUNE = tf.data.experimental.AUTOTUNE
        self.WIDTH, self.HEIGHT = image_dims

        # Download Config
        download_config = tfds.download.DownloadConfig(
            manual_dir=source_data_dir
        )

        download_and_prepare_kwargs = {
            'download_config': download_config,
        }

        # TFDS Data Loader (This step also performs dataset conversion to TFRecord)
        self.dataset, self.info = tfds.load(
            'imagenet2012',
            data_dir=os.path.join(source_data_dir, 'data'),
            split=split,
            shuffle_files=True,
            download=True,
            as_supervised=True,
            with_info=True,
            download_and_prepare_kwargs=download_and_prepare_kwargs
        )

    def preprocess_image(self, image, label):
        """
        preprocess_image

        Process the image and label to perform the following operations:
        - Min Max Scale the Image (Divide by 255)
        - Convert the numerical values of the lables to One Hot Encoded Format
        - Resize the image to 224, 224

        Args:
            image (Image Tensor): Raw Image
            label (Tensor): Numeric Labels 1, 2, 3, ...
        Returns:
            tuple: Scaled Image, One-Hot Encoded Label
        """
        image = tf.cast(image, tf.uint8)
        image = tf.image.resize(image, [self.HEIGHT, self.WIDTH])
        image = image / tf.math.reduce_max(image)
        label = tf.one_hot(indices=label, depth=self.NUM_CLASSES)
        return image, label

    @tf.function
    def augment_batch(self, image, label) -> tuple:
        """
        augment_batch
        Image Augmentation for Training:
        - Random Contrast
        - Random Brightness
        - Random Hue (Color)
        - Random Saturation
        - Random Horizontal Flip
        - Random Reduction in Image Quality
        - Random Crop
        Args:
            image (Tensor Image): Raw Image
            label (Tensor): Numeric Labels 1, 2, 3, ...
        Returns:
            tuple: Augmented Image, Numeric Labels 1, 2, 3, ...
        """
        if tf.random.normal([1]) < 0:
            image = tf.image.random_contrast(image, 0.2, 0.9)
        if tf.random.normal([1]) < 0:
            image = tf.image.random_brightness(image, 0.2)
        if self.NUM_CHANNELS == 3 and tf.random.normal([1]) < 0:
            image = tf.image.random_hue(image, 0.3)
        if self.NUM_CHANNELS == 3 and tf.random.normal([1]) < 0:
            image = tf.image.random_saturation(image, 0, 15)

        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_jpeg_quality(image, 10, 100)

        return image, label

    def get_dataset_size(self) -> int:
        """
        get_dataset_size
        Get the Dataset Size (Number of Images)
        Returns:
            int: Total Number of images in Dataset
        """
        return len(self.dataset)

    def get_num_steps(self) -> int:
        """
        get_num_steps
        Get the Number of Steps Required per Batch for Training
        Raises:
            AssertionError: Dataset Generator needs to be Initialized First
        Returns:
            int: Number of Steps Required for Training Per Batch
        """
        if self.BATCH_SIZE is None:
            raise AssertionError(
                f"Batch Size is not Initialized. Call this method only after calling: {self.dataset_generator}"
            )
        num_steps = self.get_dataset_size() // self.BATCH_SIZE + 1
        return num_steps

    def dataset_generator(self, batch_size=32, augment=False):
        """
        dataset_generator
        Create the Data Loader Pipeline and Return a Generator to Generate Datsets
        Args:
            batch_size (int, optional): Batch Size. Defaults to 32.
            augment (bool, optional): Enable/Disable Augmentation. Defaults to False.
        Returns:
            Tf.Data Generator: Dataset Generator
        """
        self.BATCH_SIZE = batch_size

        dataset = self.dataset.apply(tf.data.experimental.ignore_errors())

        dataset = dataset.shuffle(batch_size * 10)
        dataset = dataset.repeat()

        if augment:
            dataset = dataset.map(self.augment_batch, num_parallel_calls=self.AUTOTUNE)

        dataset = dataset.map(self.preprocess_image, num_parallel_calls=self.AUTOTUNE)

        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=self.AUTOTUNE)

        return dataset

    def visualize_batch(self, augment=True) -> None:
        """
        visualize_batch
        Dataset Sample Visualization
        - Supports Augmentation
        - Automatically Adjusts for Grayscale Images
        Args:
            augment (bool, optional): Enable/Disable Augmentation. Defaults to True.
        """
        if self.NUM_CHANNELS == 1:
            cmap = "gray"
        else:
            cmap = "viridis"

        dataset = self.dataset_generator(batch_size=36, augment=augment)
        image_batch, label_batch = next(iter(dataset))
        image_batch, label_batch = (
            image_batch.numpy(),
            label_batch.numpy(),
        )

        for n in range(len(image_batch)):
            ax = plt.subplot(6, 6, n + 1)
            plt.imshow(image_batch[n], cmap=cmap)
            plt.title(np.argmax(label_batch[n]))
            plt.axis("off")
        plt.show()

# Create AlexNet Model (Fused)
alexnet = create_alexnet_model(input_shape=(224, 224, 3), num_classes=1000)

metrics = [
    tf.keras.metrics.CategoricalAccuracy(),
    tf.keras.metrics.FalseNegatives(),
    tf.keras.metrics.FalsePositives(),
    tf.keras.metrics.Precision(),
    tf.keras.metrics.Recall(),
]

tf.keras.mixed_precision.set_global_policy("mixed_float16")

# Constants
BATCH_SIZE = 512
EPOCHS = 50

# Init Data Loaders
train_data_loader = ImageNetDataLoader(split=tfds.Split.TRAIN)
val_data_loader = ImageNetDataLoader(split=tfds.Split.VALIDATION)
test_data_loader = ImageNetDataLoader(split=tfds.Split.TEST)

train_generator = train_data_loader.dataset_generator(batch_size=BATCH_SIZE, augment=False)
val_generator = val_data_loader.dataset_generator(batch_size=BATCH_SIZE, augment=False)
test_generator = test_data_loader.dataset_generator(batch_size=BATCH_SIZE, augment=False)

train_steps = train_data_loader.get_num_steps()
val_steps = val_data_loader.get_num_steps()
test_steps = test_data_loader.get_num_steps()

# Compile & Train
alexnet.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.SGD(
        learning_rate=0.01, momentum=0.9, nesterov=False, name='SGD'
    ),
    metrics=metrics,
)

start = time.time()
history = alexnet.fit(
    epochs=EPOCHS,
    x=train_generator,
    steps_per_epoch=train_steps,
    validation_data=val_generator,
    validation_steps=val_steps
)
alexnet.save_weights('./models/weights/alexnet_imagenet')
print('Model %s trained in %ss' % (alexnet.name, time.time() - start))

alexnet.evaluate(test_generator, steps=test_steps)
