# -*- coding: utf-8 -*-
"""Computing embeddings for siamese networks example.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1d10wqO7gADCU_jnpP7-VOnDpYyEsMjYx
"""
import sys
sys.path.append("..")

from src.data.embeddings import *
from utils.common import *
from utils.distance import *
from src.model.alexnet import AlexNetModel
from tensorflow.keras import layers, Model

from tqdm import tqdm
from pathlib import Path

"""## Load dataset and pretrained model backbone

### Model

Load the pretrained model from TF Hub. When building, we pass the input size that the model expects.
"""

# create model
alexnet = AlexNetModel()
alexnet.compile()
# alexnet.summary()

train_ds, test_ds, validation_ds = AlexNetModel.x_dataset()

# load weights
# alexnet.fit(train_ds, validation_data=test_ds)
# alexnet.save_weights(get_modeldir('cifar10_alexnet1304.h5'))
# alexnet.evaluate(validation_ds)
alexnet.load_weights(get_modeldir('cifar10_alexnet1304.h5'))

# image features
embedding_model = Model(inputs=alexnet.input, outputs=alexnet.output)
# for layer in embedding_model.layers:
#     layer.trainable = False
# embedding_model.summary()

"""### Dataset

CIFAR 10 has shape 32x32 but the model expects 384x384, so we upsize the image (NOTE: this will likely lead to very bad performance, but it's because of CIFAR rather than the method itself. Consider using a dataset with higher image resolution; for first tests, stick to something available as tensorflow_dataset to speed up things a lot)
"""

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
images = np.concatenate([train_images, test_images])
labels = np.concatenate([train_labels, test_labels])
embedding_vds = tf.data.Dataset.from_tensor_slices((images, labels))
embedding_vds = embedding_vds.map(process_images_couple, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)

# Add batch and prefetch to dataset to speed up processing
BATCH_SIZE = 256
embedding_vds = embedding_vds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Dataset has keys "id" (that we ignore), "image" and "label".
# "image" has shape [BATCH_SIZE,32,32,3] and is an RGB uint8 image
# "label" has shape [BATCH_SIZE,1] and is an integer label (value between 0 and 9)

"""## Precompute embeddings for all dataset images

Since the network is frozen, to speed up training it's better to precalculate the image features for each image in the dataset and only use those values to train the siamese model.

For each image, we keep its label and the image features extracted by the model.
At the end, we save the computed embeddings as a Pandas dataframe, so they can be loaded back quickly without having to recompute them every time.

**NOTE**: Run this on a GPU-enabled runtime or it will take forever
"""

embeddings = embedding_model.predict(embedding_vds)
embedding_labels = np.concatenate([y for x, y in embedding_vds], axis=0)
embedding_labels = np.concatenate(embedding_labels).ravel()  # unwrap from single item array

NUM_CLASSES = np.unique(embedding_labels).shape[0]


"""# Validation

To validate the model, we load the validation chunk of the dataset and we feed it into the network. We don't need to repeat the preprocessing steps done to the dataset, because the preprocessing is embedded in the inference model by the `Rescaling` and `Resizing` layers we added above.

____________

## Visualizing embeddings in TensorBoard

In `metadata.tsv` file we list the labels in the same order as they appear in the embeddings list.
We write out the embeddings list as a tf.Variable initialized to the embeddings values, using TensorBoard's writers to specify the metadata file to use and the name of the tensor to display.

Additionally, in the specification of ProjectorConfig's proto message, there is the possibility to pass the values as a second .tsv file (`values.tsv`) instead than having them loaded from the checkpoint file.

I don't know which values are getting loaded at the moment, but since it works I won't change it further and keep both the .tsv and the checkpointed values.

(See https://stackoverflow.com/a/57230031/3214872)
"""

print('visualization')

def write_embeddings_for_tensorboard(image_vectors: list, labels: list, root_dir: Path):
    import csv
    from tensorboard.plugins import projector
    root_dir.mkdir(parents=True, exist_ok=True)
    with (root_dir / 'values.tsv').open('w') as fp:
        writer = csv.writer(fp, delimiter='\t')
        writer.writerows(image_vectors)

    with (root_dir / 'metadata.tsv').open('w') as fp:
        for lbl in labels:
            fp.write(f'{lbl}\n')

    image_vectors = np.asarray(image_vectors)
    embeddings = tf.Variable(image_vectors, name='embeddings')
    CHECKPOINT_FILE = str(root_dir / 'model.ckpt')
    ckpt = tf.train.Checkpoint(embeddings=embeddings)
    ckpt.save(CHECKPOINT_FILE)

    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = "embeddings/.ATTRIBUTES/VARIABLE_VALUE"
    embedding.metadata_path = 'metadata.tsv'
    embedding.tensor_path = 'values.tsv'
    projector.visualize_embeddings(root_dir, config)


NUM_SAMPLES_TO_DISPLAY = 3000
LOG_DIR = Path('../logs/logs_projection_alexnet2')

LOG_DIR.mkdir(exist_ok=True, parents=True)

write_embeddings_for_tensorboard(embeddings, embedding_labels, LOG_DIR)

# # Do the same with some of the training data, just to see if it works with that
# ds = embeddings_ds.take(NUM_SAMPLES_TO_DISPLAY).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
# _image_vectors = []
# _labels = []
# for feats_batch in tqdm(ds):
#     ims, lbls = feats_batch
#     ims = ims.numpy()
#     lbls = lbls.numpy()
#     embs = projection_model(ims).numpy()
#     _image_vectors.extend(embs.tolist())
#     _labels.extend(lbls.tolist())
# write_embeddings_for_tensorboard(_image_vectors, _labels, LOG_DIR / 'train')