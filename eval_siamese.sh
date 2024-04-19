#!/bin/bash

# See `man sbatch` or https://slurm.schedmd.com/sbatch.html for descriptions of sbatch options.
#SBATCH --nodes=1                   # Number of nodes to request
#SBATCH --cpus-per-task=16          # Number of CPUs to request
#SBATCH --gpus=2                    # Number of GPUs to request
#SBATCH --mem=16G                   # Amount of RAM memory requested

source /etc/bashrc.d/mamba.sh
mamba activate tf

python3 /home/astappiev/nsir/cnn_siamese.py "$@"

wait  # Wait for all jobs to complete
exit 0 # happy end
