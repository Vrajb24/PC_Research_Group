# need to change partion name correctly
CPU_PARTITION="standard"

#will add a meta file changing number of nodes for benchmarking
CORES_PER_NODE=48

# Loop through the total number of processes you want to test
for procs in 4 8 16 32 64; do
    # Calculate the number of nodes needed
    nodes=$(( (procs + CORES_PER_NODE - 1) / CORES_PER_NODE ))
    
    # Create a specific run script from the template
    run_script="run_${procs}p.slurm"
    sed -e "s/<<TOTAL_PROCS>>/${procs}/g" \
        -e "s/<<NODES>>/${nodes}/g" \
        -e "s/<<TASKS_PER_NODE>>/$((procs / nodes))/g" \
        -e "s/cpu_partition/${CPU_PARTITION}/g" \
        submit_template.slurm > ${run_script}

    echo "Submitting job for ${procs} processes on ${nodes} node(s)..."
    sbatch ${run_script}
done