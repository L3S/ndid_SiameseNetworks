#!/bin/bash

# exit on ctrl+c
trap "exit" INT

dataset=cifar10
loss=contrastive
dimensions=512
epochs=5

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate tf

cd src
for net in "alexnet" "vgg16" "mobilenet" "efficientnet" "vit" "resnet"; do
  for dataset in "cifar10" "imagenette"; do
    for loss in "contrastive" "offline-triplet" "semi-hard-triplet" "hard-triplet"; do
      for margin in "0.5" "0.75" "1" "1.25" "1.5" "1.75" "2"; do
        for epochs in 3 5 7 10 15 20 50; do
          python $net.py -ds $dataset -l $loss -m $margin -d $dimensions -e $epochs
        done
      done
    done
  done
done
