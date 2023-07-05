#!/bin/bash

# See `man sbatch` or https://slurm.schedmd.com/sbatch.html for descriptions of sbatch options.
#SBATCH --nodes=1                   # Number of nodes to request
#SBATCH --cpus-per-task=64          # Number of CPUs to request
#SBATCH --gpus=4                    # Number of GPUs to request
#SBATCH --mem=256G                  # Amount of RAM memory requested

source /opt/conda/etc/profile.d/conda.sh
conda activate tf

# print available GPUs
# python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
python3 /home/astappiev/nsir/alexnet_train.py

wait  # Wait for all jobs to complete
exit 0 # happy end
