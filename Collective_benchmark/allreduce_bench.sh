#!/bin/bash
#SBATCH --job-name=allreduce            # Job name
#SBATCH --partition=standard            # Partition (queue) name
#SBATCH --nodes=2                       # Total number of nodes
#SBATCH --ntasks-per-node=4             # MPI ranks per node
#SBATCH --time=00:30:00                 # Walltime (hh:mm:ss)
#SBATCH --output=allreduce_%j.out       # STDOUT (%j = job ID)
#SBATCH --error=allreduce_%j.err        # STDERR

# Load compiler and MPI modules
module load gcc/9.3.0 openmpi/4.0.3

# Compile the benchmark
mpicc -O3 -o allreduce_bench allreduce_bench.c

# Run the benchmark with default parameters (N=1e6, σ=65536)
srun --mpi=pmix ./allreduce_bench

# If you want to sweep N and σ, uncomment and adjust the following line:
# srun --mpi=pmix ./allreduce_bench 2097152 32768
