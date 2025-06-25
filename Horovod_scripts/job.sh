#this is just a template need to add mpi/nccl selection argument, script name, input argument values and a meta script running this job file for different combinations.
#!/bin/bash
#SBATCH --job-name=mpi_runs
#SBATCH --output=mpi_run_%A_%a.out
#SBATCH --error=mpi_run_%A_%a.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --time=00:30:00
#SBATCH --partition=standard
#SBATCH --array=0-3

# Load MPI module (might need to adjust)
module load mpi

# Define different combinations of input arguments
case $SLURM_ARRAY_TASK_ID in
    0)
        ARGS="--alpha 0.1 --beta 10"
        ;;
    1)
        ARGS="--alpha 0.1 --beta 20"
        ;;
    2)
        ARGS="--alpha 0.2 --beta 10"
        ;;
    3)
        ARGS="--alpha 0.2 --beta 20"
        ;;
    *)
        echo "Invalid task ID: $SLURM_ARRAY_TASK_ID"
        exit 1
        ;;
esac

echo "Running MPI with arguments: $ARGS"
srun --mpi=pmix_v3 ./your_program $ARGS
