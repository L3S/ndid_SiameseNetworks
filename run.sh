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
for i in {1..5}; do
  for net in "alexnet"; do
    for dataset in "imagenette"; do
      for loss in "contrastive"; do
        for margin in "1"; do
          for epochs in 5 10; do
            for dimensions in 512; do
              python $net.py -ds $dataset -l $loss -m $margin -d $dimensions -e $epochs
            done
          done
        done
      done
    done
  done
done
