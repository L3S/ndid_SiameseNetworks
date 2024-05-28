# Siamese Coding Network for Near-Duplicate Image Detection

The project structure is:
 - `/datasets` - should be downloaded from appropriate sources (CIFAR10, Imagenette, UKBench, Imagenet, etc.)
 - `/models` - computed model weights are stored here and reused if possible
 - `/data` - is where all computed vectors (embeddings) are stored
 - `/logs` - TendorBoard logs and projections

 - `/nnfaiss` - faiss analysis code
 - `/sidd` - Siamese models and training code
 - `/scripts` - extra scripts for data preparation
 - `/siftbof` - scripts for the baseline method SIFT-BoF
 - `batch.sh` - SLURM batch script for running experiments
 - `runner*.py` - SLURM entry points for experiments

All the experiments are peformed using Python 3.10 and Tensorflow 2.10+.  
On a SLURM cluster provided by L3S Research Center utilizing A100 Nvidia GPUs.

An interactive demo with real-time duplicate detection is _Work In Progress_.


## CNN Models & Weights

All the models are stored in the [model](sidd/model) directory.
Most of them, except Alexnet have Imagenet1k pre-trained weights. Alexnet was trained on Imagenet1k using [our script](scripts/train_alexnet.py).
