#!/bin/bash

# See `man sbatch` or https://slurm.schedmd.com/sbatch.html for descriptions of sbatch options.
#SBATCH --nodes=1                   # Number of nodes to request
#SBATCH --cpus-per-task=4          # Number of CPUs to request
#SBATCH --gpus=2                    # Number of GPUs to request
#SBATCH --mem=32G                   # Amount of RAM memory requested
#SBATCH --qos=standby

source /etc/bashrc.d/mamba.sh
micromamba activate tf

python3 /home/astappiev/nsir/cnn_base.py "$@"

wait  # Wait for all jobs to complete
exit 0 # happy end
