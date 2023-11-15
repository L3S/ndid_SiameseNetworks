#!/bin/bash

# exit on ctrl+c
trap "exit" INT

model=alexnet
dataset=cifar10
weights=imagenet
loss=contrastive
margin=1
epochs=5
dimensions=512

#sbatch --job-name "test-mobilenet" ./runner.sh "mobilenet" "imagenette" "contrastive" "1" "512" "5" "3105"

#DIR_NAME="margin"
#find ./faiss -maxdepth 1 -type f | xargs mv -t "./faiss/$DIR_NAME"

# Params
## margin
# for i in {1..3}; do
#   for model in "alexnet" "efficientnet" "mobilenet" "resnet" "vgg16" "vit"; do # "alexnet" "efficientnet" "mobilenet" "resnet" "vgg16" "vit"
#     for dataset in "imagenette" "cifar10"; do
#       for margin in "0.5" "0.75" "1" "1.25" "1.5" "1.75" "2"; do # "0.5" "0.75" "1" "1.25" "1.5" "1.75" "2"
#         sbatch --job-name "nsir-$model-$dataset-$loss-$margin-$dimensions-$epochs-$i" ./runner.sh "$model" "$dataset" "$weights" "$loss" "$margin" "$dimensions" "$epochs" "hpm$i"
#       done
#     done
#   done
# done

## epochs
# for i in {1..2}; do
#   for model in "alexnet" "efficientnet" "mobilenet" "resnet" "vgg16" "vit"; do # "alexnet" "efficientnet" "mobilenet" "resnet" "vgg16" "vit"
#     for dataset in "imagenette" "cifar10"; do
#       for epochs in "50"; do #  "50"
#         sbatch --job-name "nsir-$model-$dataset-$loss-$margin-$dimensions-$epochs-$i" ./runner.sh "$model" "$dataset" "$weights" "$loss" "$margin" "$dimensions" "$epochs" "hpe$i"
#       done
#     done
#   done
# done

## loss
# for i in {1..3}; do
#   for model in "alexnet" "efficientnet" "mobilenet" "resnet" "vgg16" "vit"; do # "alexnet" "efficientnet" "mobilenet" "resnet" "vgg16" "vit"
#     for dataset in "imagenette" "cifar10"; do
#       for loss in "contrastive" "easy-triplet" "semi-hard-triplet" "hard-triplet"; do
#         sbatch --job-name "nsir-$model-$dataset-$loss-$margin-$dimensions-$epochs-$i" ./runner.sh "$model" "$dataset" "$weights" "$loss" "$margin" "$dimensions" "$epochs" "hpl$i"
#       done
#     done
#   done
# done

# Projection
# for model in "alexnet" "efficientnet" "mobilenet" "resnet" "vgg16" "vit"; do
#   for dataset in "imagenette" "cifar10"; do
#     sbatch --job-name "nsir-$model-$dataset-$i" ./runner3.sh "$model" "$dataset" "proj"
#   done
# done

# Evaluate on different datasets
# weights=imagenetplus
for i in {1..3}; do
  for model in "alexnet" "efficientnet" "mobilenet" "resnet" "vgg16" "vit"; do
    for dataset in "imagenette" "cifar10"; do
      for evalds in "mirflickr"; do
        sbatch --job-name "nsir-$model-$dataset" ./runner_evals.sh "$model" "$dataset" "$evalds" "final$i"
      done
    done
  done
done

# Evaluate base CNNs
# for i in {1..3}; do
#   for model in "alexnet" "efficientnet" "mobilenet" "resnet" "vgg16" "vit"; do
#     for dataset in "cifar10"; do
#       for evalds in "mirflickr"; do # "mirflickr" "ukbench" "california" "copydays"
#         sbatch --job-name "nsir-$model-$dataset" ./runner_eval.sh "$model" "$dataset" "$evalds" "final$i"
#       done
#     done
#   done
# done
