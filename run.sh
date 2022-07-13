#!/bin/bash

# exit on ctrl+c
trap "exit" INT

dataset=cifar10
loss=contrastive
dimensions=3
epochs=5

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate tf

cd src
for margin in "0.5" "1" "1.5" "2"; do
  python alexnet.py -ds $dataset -l $loss -m $margin -d $dimensions -e $epochs
done
