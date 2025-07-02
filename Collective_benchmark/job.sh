#!/bin/bash

#SBATCH --job-name=allreduce_scaling
#SBATCH --partition=small
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=48      
#SBATCH --time=00:30:00
#SBATCH --output=allreduce_%j.out
#SBATCH --error=allreduce_%j.err
# #SBATCH --account=vrajb24

module load oneapi/mpi/latest     

#   sbatch --export=MSG_SIZE=1048576,CHUNK=256 allreduce_scaling.sbatch
: "${MSG_SIZE:=1048576}"   # default 1 MiB, override with --export
: "${CHUNK:=256}"          # default 256,    override with --export

echo "Running allreduce scaling sweep on ${SLURM_NNODES} node(s)"
echo "MSG_SIZE=${MSG_SIZE} ; CHUNK=${CHUNK}"
echo "------------------------------------------------------------------------"

################################  Benchmark loop  ############################
for P in 1 2 4 8 16 32 48 64 96; do
  # ranks-per-node ≤ 48 so they fit under the sbatch header limit
  PPN=$(( (P + SLURM_NNODES - 1) / SLURM_NNODES ))
  for i in {1..5}; do
    echo "===== Run ${i} for P=${P} ====="
    mpirun -n ${P} -ppn ${PPN} ./allreduce "${MSG_SIZE}" "${CHUNK}" \
           | tee -a run_P${P}.log
    echo
  done
done
###############################################################################
