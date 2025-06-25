#!/bin/bash
#SBATCH --job-name=allreduce_bench
#SBATCH --output=allreduce_bench_%j.out
#SBATCH --error=allreduce_bench_%j.err
#SBATCH --partition=small            # small partition (48 cores/node)
#SBATCH --nodes=1                    # single node
#SBATCH --ntasks=48                  # up to 48 MPI ranks on that node
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00              # hh:mm:ss
#SBATCH --mem-per-cpu=2G             # adjust if needed

module purge
module load gnu8/8.3.0
module load mpich/3.3

# sanity checks
which gcc
gcc --version
which mpicc
mpicc --version

# 1) Compile
mpicc -O3 -o allreduce_bench allreduce_bench.c

# 2) Benchmark parameters
N=50331648                           # total floats (~200 MiB)
P_LIST=(1 2 4 8 16 32 48)            # sweep up to 48 ranks
SIGMA_LIST=(8192 16384 32768)        # segment sizes in floats

# 3) Sweep
for P in "${P_LIST[@]}"; do
  for SIGMA in "${SIGMA_LIST[@]}"; do
    echo "===== Running P=$P, σ=$SIGMA ====="
    mpirun -np $P ./allreduce_bench $N $SIGMA \
      &>> bench_P${P}_S${SIGMA}.log
  done
done