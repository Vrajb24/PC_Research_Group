#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <unistd.h> // For usleep

// Dummy computation function
void compute(long iterations) {
    for(long i = 0; i < iterations; ++i) {
        // A simple operation to consume CPU cycles
        double result = 2.0 * i / (i + 1.0);
    }
}

#define MESSAGE_SIZE 1048576
#define NUM_CHUNKS 16
#define CHUNK_SIZE (MESSAGE_SIZE / NUM_CHUNKS)
#define COMPUTE_ITERATIONS 100000

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    // ... (MPI rank and size initialization, buffer allocation as in pipelined example) ...

    double start_time = MPI_Wtime();

    for (int i = 0; i < NUM_CHUNKS; ++i) {
        // Overlap computation with the communication of the previous chunk
        if (i > 0) {
            compute(COMPUTE_ITERATIONS);
        }

        // Initiate non-blocking communication for the current chunk
        MPI_Request reqs[2];
        MPI_Ireduce_scatter(send_buf + i * CHUNK_SIZE, temp_buf, &CHUNK_SIZE, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD, &reqs[0]);
        MPI_Iallgather(temp_buf, CHUNK_SIZE, MPI_FLOAT, recv_buf + i * CHUNK_SIZE, CHUNK_SIZE, MPI_FLOAT, MPI_COMM_WORLD, &reqs[1]);
        
        // Wait for the communication of the current chunk to complete
        MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);
    }
    
    double end_time = MPI_Wtime();
    // ... (Print timing and bandwidth results as in previous examples) ...

    MPI_Finalize();
    return 0;
}