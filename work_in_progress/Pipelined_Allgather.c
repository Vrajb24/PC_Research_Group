#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define MESSAGE_SIZE 1048576 // 1 MB
#define NUM_CHUNKS 16
#define CHUNK_SIZE (MESSAGE_SIZE / NUM_CHUNKS)

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    float* send_buf = (float*)malloc(MESSAGE_SIZE * sizeof(float));
    float* recv_buf = (float*)malloc(MESSAGE_SIZE * sizeof(float));
    float* temp_buf = (float*)malloc(CHUNK_SIZE * sizeof(float));

    // Initialize data
    for (int i = 0; i < MESSAGE_SIZE; ++i) {
        send_buf[i] = (float)rank;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    for (int i = 0; i < NUM_CHUNKS; ++i) {
        // Scatter-reduce a chunk of data
        MPI_Reduce_scatter(send_buf + i * CHUNK_SIZE, temp_buf, &CHUNK_SIZE, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        // Allgather the result of the chunk reduction
        MPI_Allgather(temp_buf, CHUNK_SIZE, MPI_FLOAT, recv_buf + i * CHUNK_SIZE, CHUNK_SIZE, MPI_FLOAT, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();

    if (rank == 0) {
        double elapsed_time = end_time - start_time;
        double bandwidth = (2.0 * MESSAGE_SIZE * sizeof(float)) / elapsed_time / 1e9; // GB/s
        printf("Pipelined Allreduce with %d processes and message size %d bytes took %f seconds. Bandwidth: %f GB/s\n",
               size, MESSAGE_SIZE * (int)sizeof(float), elapsed_time, bandwidth);
    }

    free(send_buf);
    free(recv_buf);
    free(temp_buf);

    MPI_Finalize();
    return 0;
}