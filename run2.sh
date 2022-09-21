#!/bin/bash

# exit on ctrl+c
trap "exit" INT

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate tf

cd src
for net in "alexnet" "vgg16" "mobilenet" "efficientnet" "vit" "resnet"; do
  for dataset in "cifar10" "imagenette"; do
    for loss in "contrastive" "offline-triplet"; do
        python $net.py -ds $dataset -l $loss -m 1 -d 512 -e 7
    done
  done
done
