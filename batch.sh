#!/bin/bash

# exit on ctrl+c
trap "exit" INT

model=alexnet
dataset=cifar10
weights=load
loss=contrastive
margin=1
epochs=5
dimensions=512

# Evaluate base CNNs (trained on ImageNet)
# for i in {1..3}; do
#   for model in "alexnet" "efficientnet" "mobilenet" "resnet" "vgg16" "vit"; do
#     for evalds in "mirflickr"; do # "mirflickr" "ukbench" "californiand" "copydays"
#       sbatch --job-name "cnn-$model-$dataset" ./eval_cnn.sh -CM "$model" -ED "$evalds" -s "final$i"
#     done
#   done
# done

# Train Siamese on ND datasets
for i in {1..3}; do
  for model in "alexnet" "efficientnet" "mobilenet" "resnet" "vgg16" "vit"; do
    for dataset in "copydays"; do # "ukbench", "copydays", "holidays", "californiand", "mirflickr"
      for margin in "1" "1.5" "2"; do # "1" "1.5" "2"
        for loss in "contrastive" "semi-hard-triplet" "hard-triplet"; do
          for epochs in "10" "20" "30"; do # "10" "20" "30"
            sbatch --job-name "sidd-$model-$dataset" ./eval_siamese.sh -CM $model -D $dataset -m $margin -l $loss -e $epochs -s exp$i --save-vectors True
          done
        done
      done
    done
  done
done

# Evaluate SiameseCNNs on ND datasets
# for i in {1..3}; do
#   for model in "alexnet" "efficientnet" "mobilenet" "resnet" "vgg16" "vit"; do
#     for dataset in "ukbench", "copydays", "holidays", "californiand", "mirflickr"; do
#       for margin in "1"; do
#         for loss in "semi-hard-triplet"; do
#           for epochs in "10"; do
#             for evalds in "ukbench", "copydays", "holidays", "californiand", "mirflickr"; do
#               sbatch --job-name "sidd-eval-$model-$dataset" ./eval_siamese.sh -CM $model -D $dataset -m $margin -l $loss -e $epochs -s exp$i --save-vectors True
#             done
#           done
#         done
#       done
#     done
#   done
# done
