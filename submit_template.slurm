#!/bin/bash

#SBATCH --job-name=hvd_scale_<<TOTAL_PROCS>>p
#SBATCH --output=hvd_scale_<<TOTAL_PROCS>>p_%j.out
#SBATCH --error=hvd_scale_<<TOTAL_PROCS>>p_%j.err
#SBATCH --nodes=<<NODES>>
#SBATCH --ntasks-per-node=<<TASKS_PER_NODE>>
#SBATCH --partition=cpu_partition # IMPORTANT: Replace with a real CPU partition name from 'sinfo'
#SBATCH --time=00:30:00

echo "Loading modules..."
# These are the modules we assume will be fixed and working
module purge
module load python/3.6
module load python/horovod-cpu
echo "Modules loaded."

# Get the list of hosts from Slurm
HOSTS=$(scontrol show hostnames $SLURM_JOB_NODELIST)
echo "Total processes: $SLURM_NTASKS"
echo "Running on hosts: $HOSTS"

# Execute the Horovod job
horovodrun -np $SLURM_NTASKS -H $(echo $HOSTS | sed 's/ /,/g') python horovod_scaling_test.py