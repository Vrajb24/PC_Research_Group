# Distributed Deep Learning Communication Optimization

This repository hosts the research, implementations, and benchmarks developed by our group at IIT Kanpur focused on optimizing communication between compute nodes in large-scale high-performance computing (HPC) environments. Our work targets the communication bottlenecks in distributed deep learning training, especially for large-scale models running across multi-node clusters.

## Table of Contents
- [Research Focus](#-research-focus)
- [Current Projects](#-current-projects)
- [Repository Structure](#-repository-structure)
- [Dependencies](#-dependencies)
- [Running Experiments](#-running-experiments)
- [Results](#-results)
- [Publications & Reports](#-publications--reports)
- [Contributing](#-contributing)
- [License](#-license)

## 🔬 Research Focus

Our research focuses on quantifying and mitigating the communication overhead inherent in distributed data-parallel training. We investigate the performance of collective communication primitives like `AllReduce` and `AllGather` under various workloads and network conditions. Key areas of interest include:
-   **Scaling Efficiency Analysis:** Measuring the throughput (e.g., images/second) of deep learning models as we scale from a few processes on a single node to hundreds of processes across multiple nodes.
-   **Communication Backend Comparison:** Evaluating the performance differences between various communication backends, such as MPI-based approaches (OpenMPI, MPICH) and hardware-accelerated libraries like NVIDIA's NCCL for GPU clusters.
-   **Topology-Aware Communication:** Developing strategies that take the physical network topology of the HPC cluster into account to optimize data exchange patterns.

## 🧠 Current Projects

1.  **Horovod Scaling Efficiency Benchmark:** An experiment designed to measure the weak and strong scaling performance of a ResNet-18 model on the CIFAR-10 dataset. This project helps us understand how communication overhead impacts total throughput as we increase the number of compute processes.
2.  **Hyperparameter Sweep Framework:** A generic framework using Slurm job arrays to run an executable (e.g., a compiled C++/MPI program) with multiple combinations of command-line arguments. This is useful for exploring the parameter space of custom communication algorithms.
3.  **Microbenchmarks (Ongoing):** Development of low-level benchmarks to measure the latency and bandwidth of fundamental collective operations (`AllReduce`, `Scatter-Gather`, etc.) independent of a full model training workload.
## ⚙️ Dependencies

-   **Frameworks:**
    -   PyTorch (>= 1.8.0)
    -   torchvision
    -   Horovod (>= 0.21.0)
-   **System:**
    -   A Slurm-based HPC environment
    -   An MPI implementation (e.g., OpenMPI, MPICH) compatible with the cluster
    -   For GPU benchmarks: NVIDIA Drivers, CUDA Toolkit (>= 11.0), NCCL2

## 🚀 Running Experiments

### 1. Horovod Scaling Test

This test runs the `horovod_scaling_test.py` script across a varying number of processes (e.g., 4, 8, 16, 32, 64) and measures the training throughput.

1.  Navigate to the `scripts` directory.
2.  Open the `run_scaling_test.sh` script and modify the following variables at the top:
    -   `CPU_PARTITION`: Set this to the name of the CPU partition on your Slurm cluster (e.g., "standard", "compute").
    -   `CORES_PER_NODE`: Set this to the number of physical cores on a single compute node.
3.  Execute the script:
    ```bash
    bash run_scaling_test.sh
    ```
This will automatically generate Slurm scripts for each process count, submit them to the queue, and save the output logs in the `scripts` directory.

### 2. Hyperparameter Sweep

This uses a Slurm job array to run a program multiple times with different arguments.

1.  Open `scripts/submit_hyperparam_sweep.slurm`.
2.  Modify the `case` statement to include the argument combinations you wish to test.
3.  Change the final `srun` line from `./your_program` to point to your actual compiled executable.
4.  Submit the job to Slurm:
    ```bash
    sbatch scripts/submit_hyperparam_sweep.slurm
    ```

## 📊 Results

After running the scaling test, you can quickly aggregate the results using `grep`.

```bash
# From the scripts/ directory, after jobs have completed
grep "Global Img/sec" *.out