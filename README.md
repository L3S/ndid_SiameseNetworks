# Siamese Coding Network for Near-Duplicate Image Detection

The project structure is:
 - `/datasets` - datasets dir, should be downloaded from appropriate sources (CIFAR10, Imagenette, UKBench, Imagenet1k)
 - `/models` - computed model weights are stored here and reused if possible
 - `/data` - is where all computed vectors are stored
 - `/nnfaiss` - faiss analysis code
 - `/sidd` - Siamese models and training code
 - `/scripts` - extra scripts for data preparation
 - `batch.sh` - SLURM batch script for running experiments
 - `runner*.py` - SLURM entry points for experiments

All the experiments are made with Python 3.10 and Tensorflow 2.10+.  
On a SLURM cluster provided by L3S Research Center utilizing A100 Nvidia GPUs.

An interactive demo with real-time duplicate detection is _Work In Progress_.
