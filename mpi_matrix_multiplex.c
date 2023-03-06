#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int n = 1024;  // matrix dimension
    int block_size, num_procs, rank;
    double *A, *B, *C;
    double *subA, *subB, *subC;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    block_size = n / num_procs;  // sub-matrix dimension
    subA = (double*)malloc(block_size*block_size*sizeof(double));
    subB = (double*)malloc(block_size*block_size*sizeof(double));
    subC = (double*)malloc(block_size*block_size*sizeof(double));
    if (rank == 0) {
        A = (double*)malloc(n*n*sizeof(double));
        B = (double*)malloc(n*n*sizeof(double));
        C = (double*)calloc(n*n, sizeof(double));
        // Initialize matrices A and B
        for (int i = 0; i < n*n; i++) {
            A[i] = i % n;
            B[i] = i % n;
        }
    }
    // Scatter sub-matrices of A and B to each process
    MPI_Scatter(A, block_size*block_size, MPI_DOUBLE, subA, block_size*block_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(B, block_size*block_size, MPI_DOUBLE, subB, block_size*block_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // Compute sub-matrix multiplication
    for (int i = 0; i < block_size; i++) {
        for (int j = 0; j < block_size; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += subA[i*n+k] * subB[k*block_size+j];
            }
            subC[i*block_size+j] = sum;
        }
    }
    // Gather sub-matrices of C to the root process
    MPI_Gather(subC, block_size*block_size, MPI_DOUBLE, C, block_size*block_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        // Print the result matrix C
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                printf("%f ", C[i*n+j]);
            }
            printf("\n");
        }
        free(A);
        free(B);
        free(C);
    }
    free(subA);
    free(subB);
    free(subC);
    MPI_Finalize();
    return 0;
}
