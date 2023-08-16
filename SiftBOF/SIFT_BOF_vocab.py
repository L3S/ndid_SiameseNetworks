# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 08:43:28 2022

@author: franziska

originally taken from https://www.codeproject.com/articles/619039/bag-of-features-descriptor-on-sift-features-with-o
"""
import cv2 as cv
import numpy as np
import bz2
import pickle
import tensorflow as tf
import tensorflow_datasets as tfds
from glob import glob


SIFT = cv.SIFT_create()


def save_vectors(values, names, labels, path):
    data = [values,names,labels]

    with bz2.BZ2File(path, 'wb') as f:
        pickle.dump(data, f, 4)


def read_cifar(path): 
    """
        
        - Adapted from /sidd/data/cifar.py

    """
    train_ods = tf.keras.utils.image_dataset_from_directory(
            directory=path + 'train/',
            labels='inferred',
            label_mode='int',
            image_size=(32,32),
            interpolation='nearest',
            batch_size = None
        )
    return train_ods


def read_imagenette(path):
    """
        
        - Adapted from /sidd/data/imagenette.py

    """
    train_ods = tf.keras.utils.image_dataset_from_directory(
            directory=path + 'train/',
            labels='inferred',
            label_mode='int',
            image_size=(400, 320),
            interpolation='nearest',
            batch_size = None
        )
    return train_ods

def read_ukbench(path):
    """
        
        - Adapted from /sidd/data/ukbench.py

    """
    def load(path):
            image_raw = tf.io.read_file(path)
            decoded = tf.image.decode_jpeg(image_raw, channels=0)
            resized = tf.image.resize(decoded, [244,244], method='nearest')

            label = tf.strings.split(path, 'ukbench')[2]
            label = tf.strings.split(label, '.', 1)[0]
            label = tf.strings.to_number(label, tf.int32)
            return resized, label
    def train(image, label):
        return label%4!=0
    def test(image, label):
        return label%4==0
    dataset_path_glob = glob(path + '*.jpg')
    ds = tf.data.Dataset.from_tensor_slices(dataset_path_glob).map(load)
    print(len(ds), flush=True)
    train_ds = ds.filter(train)
    #test_ds = ds.filter(test)
    return train_ds


def read_ImageNet1k(path):
    """
        
        - Adapted from /sidd/data/ukbench.py

    """
    print(path, flush=True)
    builder = tfds.builder('imagenet2012', data_dir=path+'data/')
    #print(builder.info)

    download_config = tfds.download.DownloadConfig(
        manual_dir=path
    )

    builder.download_and_prepare(download_config=download_config)

    train_ds = builder.as_dataset(split=tfds.Split.TRAIN, as_supervised=True)
    test_ds = builder.as_dataset(split=tfds.Split.TEST, as_supervised=True)
    val_ds = builder.as_dataset(split=tfds.Split.VALIDATION, as_supervised=True)
    assert isinstance(train_ds, tf.data.Dataset)
    assert isinstance(test_ds, tf.data.Dataset)
    assert isinstance(val_ds, tf.data.Dataset)
    print(len(train_ds), len(val_ds), len(test_ds), flush=True)
    ds = train_ds.concatenate(val_ds).concatenate(test_ds)
    print(len(ds), flush=True)
    return ds


def get_keypoints(img, desc=True):
    if desc == True:
        keypoints, descriptors = SIFT.detectAndCompute(img,None)
        if len(keypoints)==0:
            return None, None
        return  keypoints, descriptors
    else: 
        keypoints = SIFT.detect(img, None)
        return keypoints              
    

def read_dataset(path, dataset):
    if dataset == 'cifar10':
        path+='/cifar10/'
        print("reading data from " + path)
        images = read_cifar(path)
    if dataset == 'imagenette':
        path+= '/imagenette2/'
        print("reading data from " + path)
        images = read_imagenette(path)
    if dataset == 'ukbench':
        
        path+= '/ukbench/'
        print("reading data from " + path)
        images = read_ukbench(path)
    if dataset == 'imagenet':
        path += '/imagenet/'
        images = read_ImageNet1k(path)
    
    
    return images


def preprocess_image(image, label):
    
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
            
            
            - Adapted from /sidd/data/imagenet1k.py
        """
        
        image = tf.image.resize(image, [244,244])
        image = tf.cast(image, tf.uint8)
        
        return image, label



if __name__ == '__main__':
    

    input_folder = "../Images"
    output_folder = ""
    # Datafile
    dataset = 'imagenette'
    
    dictionary_size = 512
    
    ds = read_dataset(input_folder, dataset)

    ds =ds.map(preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    stop_criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100,0.2) 
    retries = 5
    flags = cv.KMEANS_PP_CENTERS
    
    bowTrainer = cv.BOWKMeansTrainer(dictionary_size, stop_criteria, retries, flags)
    i = 0
    for dat in ds:
        image = dat[0].numpy()
        image=cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        keypoints, descriptors = get_keypoints(image, desc=True)
        if type(descriptors) != type(None):
            bowTrainer.add(descriptors)
        if i%1000 == 0:
            print(i, flush=True)
        i+= 1


    print('Added ' + str(i) + ' images to the trainer.')

    
    #plt.imshow(image, cmap='gray')
    print("start clustering", flush=True)
    dictionary = bowTrainer.cluster()
    print("dictionary clustered", flush=True)
    outfile_name = dataset + '_d'+ str(dictionary_size)+ '_BOF_vocab'
    np.save(output_folder + outfile_name, dictionary)
    print('dictionary saved: ', outfile_name, flush=True)
