#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Each process sends a different amount of data
    int send_count = (rank + 1) * 1024; // Example of varying sizes
    float* send_buf = (float*)malloc(send_count * sizeof(float));

    // Initialize data
    for (int i = 0; i < send_count; ++i) {
        send_buf[i] = (float)rank;
    }

    int* recv_counts = (int*)malloc(size * sizeof(int));
    int* displs = (int*)malloc(size * sizeof(int));

    // Gather the send counts from all processes
    MPI_Allgather(&send_count, 1, MPI_INT, recv_counts, 1, MPI_INT, MPI_COMM_WORLD);

    // Calculate displacements for MPI_Allgatherv
    displs[0] = 0;
    int total_recv_count = recv_counts[0];
    for (int i = 1; i < size; ++i) {
        displs[i] = displs[i-1] + recv_counts[i-1];
        total_recv_count += recv_counts[i];
    }

    float* recv_buf = (float*)malloc(total_recv_count * sizeof(float));

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    MPI_Allgatherv(send_buf, send_count, MPI_FLOAT, recv_buf, recv_counts, displs, MPI_FLOAT, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();

    if (rank == 0) {
        double elapsed_time = end_time - start_time;
        long total_bytes = (long)total_recv_count * sizeof(float);
        double bandwidth = (double)total_bytes / elapsed_time / 1e9; // GB/s
        printf("Segmented Allgather with %d processes took %f seconds. Total data gathered: %ld bytes. Bandwidth: %f GB/s\n",
               size, elapsed_time, total_bytes, bandwidth);
    }

    free(send_buf);
    free(recv_buf);
    free(recv_counts);
    free(displs);

    MPI_Finalize();
    return 0;
}