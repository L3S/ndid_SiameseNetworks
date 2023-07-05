import os

import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from ndid.data import AsbDataset
from ndid.utils.common import get_dataset

plt.rcParams["figure.figsize"] = 30, 30


class ImageNet1k(AsbDataset):
    def __init__(self, map_fn=None, batch_size=None):
        super(ImageNet1k, self).__init__(name='imagenet1k', num_classes=1000, batch_size=batch_size, map_fn=map_fn)

    def _load_dataset(self, image_size, batch_size, map_fn):
        builder = tfds.builder('imagenet2012')

        download_config = tfds.download.DownloadConfig(
            manual_dir=str(get_dataset('imagenet'))
        )

        builder.download_and_prepare(download_config=download_config)

        train_ds = builder.as_dataset(split=tfds.Split.TRAIN, batch_size=batch_size, as_supervised=True)
        test_ds = builder.as_dataset(split=tfds.Split.TEST, batch_size=batch_size, as_supervised=True)
        val_ds = builder.as_dataset(split=tfds.Split.VALIDATION, batch_size=batch_size, as_supervised=True)
        assert isinstance(train_ds, tf.data.Dataset)
        assert isinstance(test_ds, tf.data.Dataset)
        assert isinstance(val_ds, tf.data.Dataset)
        return train_ds, val_ds, test_ds


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
