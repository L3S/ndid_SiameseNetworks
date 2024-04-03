#!/bin/bash

# See `man sbatch` or https://slurm.schedmd.com/sbatch.html for descriptions of sbatch options.
#SBATCH --nodes=1                   # Number of nodes to request
#SBATCH --cpus-per-task=16          # Number of CPUs to request
#SBATCH --gpus=2                    # Number of GPUs to request
#SBATCH --mem=16G                   # Amount of RAM memory requested

source /opt/conda/etc/profile.d/conda.sh
conda activate tf

python3 /home/astappiev/nsir/cnn.py -M "$1" -D "$2" -s "$3" --ukbench True --cnn-vectors True --project-vectors True

wait  # Wait for all jobs to complete
exit 0 # happy end
