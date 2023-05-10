#!/bin/bash

# exit on ctrl+c
trap "exit" INT

model=alexnet
dataset=cifar10
loss=contrastive
margin=1
epochs=5
dimensions=512

for i in {1..5}; do
  for model in "alexnet"; do # "alexnet" "efficientnet" "mobilenet" "resnet" "vgg16" "vit"
    for dataset in "imagenette" "cifar10"; do
      for loss in "contrastive"; do # "contrastive" "easy-triplet" "semi-hard-triplet" "hard-triplet"
        for margin in "1"; do # "0.5" "0.75" "1" "1.25" "1.5" "1.75" "2"
          for epochs in 5 10; do
            for dimensions in 512; do
              sbatch --job-name "nsir-$model-$dataset-$loss-$margin-$dimensions-$epochs-$i" ./runner.sh "$model" "$dataset" "$loss" "$margin" "$dimensions" "$epochs" "2604$i"
            done
          done
        done
      done
    done
  done
done
