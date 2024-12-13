#!/bin/bash
#SBATCH --job-name="hyena_pretrain_test"
#SBATCH --output="a.out.%j.%N.out"
#SBATCH --partition=gpuA40x4
#SBATCH --mem=128G
#SBATCH --nodes=2                     # Request 2 nodes
#SBATCH --ntasks-per-node=1            # 1 task per node (since we use 1 GPU per node)
#SBATCH --cpus-per-task=4              # Allocate 4 CPUs per task
#SBATCH --gpus-per-node=4              # 1 GPU per node
#SBATCH --gpu-bind=closest             # Bind CPUs closest to the GPUs
#SBATCH --account=bdhi-delta-gpu       # Replace with your actual account name
#SBATCH -t 30:00:00                    # Adjust time as needed
#SBATCH -e slurm-%j.err                # Standard error log
#SBATCH -o slurm-%j.out                # Standard output log
#SBATCH --mail-type=BEGIN # For when the job starts
#SBATCH --mail-type=END   # For when the job ends
#SBATCH --mail-type=FAIL  # For when the job fails
#SBATCH --mail-user=pablo.maldonado@utah.edu 

# Set OpenMP to use 1 thread per process
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # Optimizing memory allocation

# Load necessary modules 
module load nvidia/24.5
module load python/3.11.6  # Ensure the correct Python version
module load cuda/11.8.0    # Ensure the correct CUDA version, adjust if necessary

# Activate the environment
source $(poetry env info --path)/bin/activate 

# Login to WandB if not done globally
wandb login 7194520caeea29f37cac5c0c29b0432cb3f9b59c

# Run the training script using srun with the required number of GPUs and CPUs
srun --mpi=pmi2 python3 -m train \
   wandb.project=hyena-dna-multispecies-32k \
   wandb.entity=gregg_lab \
   experiment=multispecies/multispecies_hyena \
   model.d_model=256 \
   model.n_layer=4 \
   dataset.batch_size=16 \
   train.global_batch_size=64 \
   dataset.max_length=32002 \
   optimizer.lr=3e-4 \
   trainer.devices=4 \
   trainer.num_nodes=2 \
   trainer.strategy=ddp \
   trainer.accelerator=gpu

# Notes:
# 1. Ensure your WandB API key is set up and your environment is configured properly. use gregg_lab to send logs to team
# 2. `trainer.num_nodes=2` and `trainer.strategy=ddp` flags ensure multi-node training. changed to 8 to train larger, longer sequence model
# 3. `trainer.accelerator=gpu` sets the correct device for training. make sure the strategy is set
# 4. 