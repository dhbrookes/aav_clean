#!/bin/bash
#SBATCH --cluster=beef            # Don't change
#SBATCH --partition=long          # Don't change
#SBATCH --account=researcher      # Don't change
#SBATCH --job-name=aav5_optimize
#SBATCH --output=aav5_post_pre_process.out
#SBATCH --gres=gpu:0              # Number of GPU(s) per node.
#SBATCH --cpus-per-task=8         # CPU cores/threads
#SBATCH --mem=10000M               # memory per node
#SBATCH --time=0-48:00            # Max time (DD-HH:MM)
#SBATCH --ntasks=1                # Only set to >1 if you want to use multi-threading
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

python ../src/pre_process.py old_nnk pre -r -c -p 2
python ../src/pre_process.py old_nnk post -r -c -p 2

# python ../src/pre_process.py new_nnk pre -r -c -p 2
# python ../src/pre_process.py new_nnk post -r -c -p 2

# python ../src/pre_process.py lib_b pre -r -c -p 2
# python ../src/pre_process.py lib_b post -r -c -p 2

# python ../src/pre_process.py lib_c pre -r -c -p 2
# python ../src/pre_process.py lib_c post -r -c -p 2