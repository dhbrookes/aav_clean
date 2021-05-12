#!/bin/bash
#SBATCH --cluster=beef            # Don't change
#SBATCH --partition=long          # Don't change
#SBATCH --account=researcher      # Don't change
#SBATCH --job-name=aav5_ann
#SBATCH --output=aav5_ann_slurm.out
#SBATCH --gres=gpu:1              # Number of GPU(s) per node.
#SBATCH --cpus-per-task=8         # CPU cores/threads
#SBATCH --mem=100000M               # memory per node
#SBATCH --time=0-96:00            # Max time (DD-HH:MM)
#SBATCH --ntasks=1                # Only set to >1 if you want to use multi-threading
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

python ../src/keras_models_run.py old_nnk ann -d 100 -e is
# python ../src/keras_models_run.py old_nnk ann -d 200 -e is
# python ../src/keras_models_run.py old_nnk ann -d 500 -e is
# python ../src/keras_models_run.py old_nnk ann -d 1000 -e is
# python ../src/keras_models_run.py old_nnk ann -d 200 -u -e is
