#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Pipelined ring Allreduce */
void run_pipelined(float *buf, int N, int P, int rank) {
    int chunk = N / P;
    float *tmp = malloc(chunk * sizeof(float));
    MPI_Request reqs[2];
    MPI_Status stats[2];
    int left = (rank - 1 + P) % P;
    int right = (rank + 1) % P;

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    /* 1) Scatter-reduce phase */
    for (int step = 0; step < P - 1; ++step) {
        int send_idx = (rank - step + P) % P;
        int recv_idx = (rank - step - 1 + P) % P;
        /* post nonblocking receive from left neighbor */
        MPI_Irecv(tmp, chunk, MPI_FLOAT, left, 0, MPI_COMM_WORLD, &reqs[0]);
        /* post nonblocking send of my current chunk to right neighbor */
        MPI_Isend(buf + send_idx * chunk, chunk, MPI_FLOAT, right, 0,
                  MPI_COMM_WORLD, &reqs[1]);
        MPI_Waitall(2, reqs, stats);
        /* element-wise reduction */
        float *recv_ptr = buf + recv_idx * chunk;
        for (int i = 0; i < chunk; ++i) {
            recv_ptr[i] += tmp[i];
        }
    }

    /* 2) Allgather phase */
    for (int step = 0; step < P - 1; ++step) {
        int send_idx = (rank - step - 1 + P) % P;
        int recv_idx = (rank - step - 2 + P) % P;
        MPI_Irecv(buf + recv_idx * chunk, chunk, MPI_FLOAT, left, 1,
                  MPI_COMM_WORLD, &reqs[0]);
        MPI_Isend(buf + send_idx * chunk, chunk, MPI_FLOAT, right, 1,
                  MPI_COMM_WORLD, &reqs[1]);
        MPI_Waitall(2, reqs, stats);
    }

    double t1 = MPI_Wtime();
    if (rank == 0)
        printf("Pipelined Allreduce time: %f seconds\n", t1 - t0);

    free(tmp);
}

/* Segmented ring Allreduce */
void run_segmented(float *buf, int N, int P, int rank, int sigma) {
    int chunk = N / P;
    int nseg = chunk / sigma;
    float *tmp = malloc(sigma * sizeof(float));
    MPI_Request reqs[2];
    MPI_Status stats[2];
    int left = (rank - 1 + P) % P;
    int right = (rank + 1) % P;

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    /* 1) Scatter-reduce phase (segment by segment) */
    for (int s = 0; s < nseg; ++s) {
        for (int step = 0; step < P - 1; ++step) {
            int send_chunk = (rank - step + P) % P;
            int recv_chunk = (rank - step - 1 + P) % P;
            float *send_ptr = buf + send_chunk * chunk + s * sigma;
            float *recv_ptr = buf + recv_chunk * chunk + s * sigma;

            MPI_Irecv(tmp, sigma, MPI_FLOAT, left, 10, MPI_COMM_WORLD, &reqs[0]);
            MPI_Isend(send_ptr, sigma, MPI_FLOAT, right, 10, MPI_COMM_WORLD,
                      &reqs[1]);
            MPI_Waitall(2, reqs, stats);

            for (int i = 0; i < sigma; ++i) {
                recv_ptr[i] += tmp[i];
            }
        }
    }

    /* 2) Allgather phase (segment by segment) */
    for (int s = 0; s < nseg; ++s) {
        for (int step = 0; step < P - 1; ++step) {
            int send_chunk = (rank - step - 1 + P) % P;
            int recv_chunk = (rank - step - 2 + P) % P;
            float *send_ptr = buf + send_chunk * chunk + s * sigma;
            float *recv_ptr = buf + recv_chunk * chunk + s * sigma;

            MPI_Irecv(recv_ptr, sigma, MPI_FLOAT, left, 20, MPI_COMM_WORLD,
                      &reqs[0]);
            MPI_Isend(send_ptr, sigma, MPI_FLOAT, right, 20, MPI_COMM_WORLD,
                      &reqs[1]);
            MPI_Waitall(2, reqs, stats);
        }
    }

    double t1 = MPI_Wtime();
    if (rank == 0)
        printf("Segmented Allreduce time (σ=%d): %f seconds\n", sigma, t1 - t0);

    free(tmp);
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int P, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &P);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* Default problem size and segment size; allow override via args */
    int N = 1024 * 1024;
    int sigma = 65536;
    if (argc >= 2) N     = atoi(argv[1]);
    if (argc >= 3) sigma = atoi(argv[2]);

    if (N % P != 0 || (N / P) % sigma != 0) {
        if (rank == 0)
            fprintf(stderr,
                    "Error: N must be divisible by P, and (N/P) by sigma.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    float *buf = malloc(N * sizeof(float));
    /* initialize buffer to rank (so we can verify correctness if desired) */
    for (int i = 0; i < N; ++i) {
        buf[i] = (float)rank;
    }

    run_pipelined(buf, N, P, rank);
    run_segmented(buf, N, P, rank, sigma);

    free(buf);
    MPI_Finalize();
    return 0;
}
