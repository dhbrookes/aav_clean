#!/bin/bash
#SBATCH --cluster=beef            # Don't change
#SBATCH --partition=long          # Don't change
#SBATCH --account=researcher      # Don't change
#SBATCH --job-name=aav5_optimize
#SBATCH --output=opt_results_nuc_15.out
#SBATCH --gres=gpu:0              # Number of GPU(s) per node.
#SBATCH --cpus-per-task=8         # CPU cores/threads
#SBATCH --mem=10000M               # memory per node
#SBATCH --time=0-48:00            # Max time (DD-HH:MM)
#SBATCH --ntasks=1                # Only set to >1 if you want to use multi-threading
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

python ../src/entropy_opt.py ../models/old_nnk_ann_100_is opt_results_nuc_15.npy --min_lambda 1 --max_lambda 2.5 --num_lambda 150 --num_iter 3000 --learning_rate 0.01 --num_samples 1000  --encoding is
