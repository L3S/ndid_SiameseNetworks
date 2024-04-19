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
#         sbatch --job-name "sidd-$model-$dataset" ./eval_cnn.sh -M "$model" -D "$dataset" -ED "$evalds" -s "final$i"
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
#         sbatch --job-name "sidd-$model-$dataset" ./eval_siamese.sh -M "$model" -D "$dataset" -ED "$evalds" -s "final$i"
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
              sbatch --job-name "sidd-$model-$dataset" ./eval_siamese.sh -M $model -D $datasetnch -m $margin -l $loss -e $epochs -s exp$i --save-vectors True
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
#         sbatch --job-name "sidd-$model-$dataset-$loss-$margin-$dimensions-$epochs-$i" ./runner.sh -M "$model" -D "$dataset" -W "$weights" -l "$loss" -m "$margin" -d "$dimensions" -e "$epochs" -s "hpm$i"
#       done
#     done
#   done
# done
