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

# Evaluate base CNNs
# for i in {1..3}; do
#   for model in "alexnet" "efficientnet" "mobilenet" "resnet" "vgg16" "vit"; do
#     for dataset in "cifar10"; do
#       for evalds in "mirflickr"; do # "mirflickr" "ukbench" "californiand" "copydays"
#         sbatch --job-name "nsir-$model-$dataset" ./eval_cnn.sh "$model" "$dataset" "$evalds" "final$i"
#       done
#     done
#   done
# done

# Evaluate SiameseCNNs
# weights=imagenetplus
# for i in {1..3}; do
#   for model in "alexnet" "efficientnet" "mobilenet" "resnet" "vgg16" "vit"; do
#     for dataset in "imagenette" "cifar10"; do
#       for evalds in "mirflickr"; do
#         sbatch --job-name "nsir-$model-$dataset" ./eval_siamese.sh "$model" "$dataset" "$evalds" "final$i"
#       done
#     done
#   done
# done

# Train and evaluate Siamese on ND datasets
for i in {1..3}; do
  for model in "alexnet" "efficientnet" "mobilenet" "resnet" "vgg16" "vit"; do
    for dataset in "imagenette"; do
      for evalds in "ukbench" "californiand"; do # "mirflickr"
        for margin in "1" "1.5" "2"; do
          for loss in "contrastive" "semi-hard-triplet" "hard-triplet"; do
            for epochs in "10" "20" "30"; do
              sbatch --job-name "nsir-$model-$dataset" ./eval_cnn.sh "$model" "$dataset" "$evalds" "final$i"
            done
          done
        done
      done
    done
  done
done

# All params
# for i in {1..3}; do
#   for model in "alexnet" "efficientnet" "mobilenet" "resnet" "vgg16" "vit"; do
#     for dataset in "imagenette" "cifar10"; do
#       for margin in "0.5" "0.75" "1" "1.25" "1.5" "1.75" "2"; do # "0.5" "0.75" "1" "1.25" "1.5" "1.75" "2"
#         sbatch --job-name "nsir-$model-$dataset-$loss-$margin-$dimensions-$epochs-$i" ./runner.sh "$model" "$dataset" "$weights" "$loss" "$margin" "$dimensions" "$epochs" "hpm$i"
#       done
#     done
#   done
# done
