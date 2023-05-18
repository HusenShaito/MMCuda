#include <stdio.h>
#include <time.h>
#define M 1000
#define N 1000
#define K 1000
#define TILE_SIZE 32

__global__ void matrixMulTiled(int *A, int *B, int *C, int m, int n, int k) {
    __shared__ int shared_A[TILE_SIZE][TILE_SIZE];
    __shared__ int shared_B[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    int sum = 0;

    for (int t = 0; t < (n + TILE_SIZE - 1) / TILE_SIZE; t++) {
        if (row < m && t * TILE_SIZE + tx < n)
            shared_A[ty][tx] = A[row * n + t * TILE_SIZE + tx];
        else
            shared_A[ty][tx] = 0;

        if (t * TILE_SIZE + ty < n && col < k)
            shared_B[ty][tx] = B[(t * TILE_SIZE + ty) * k + col];
        else
            shared_B[ty][tx] = 0;

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; i++)
            sum += shared_A[ty][i] * shared_B[i][tx];

        __syncthreads();
    }

    if (row < m && col < k)
        C[row * k + col] = sum;
}

int main() {
    int *A, *B, *C;
    int *d_A, *d_B, *d_C;
    int sizeA = M * N * sizeof(int);
    int sizeB = N * K * sizeof(int);
    int sizeC = M * K * sizeof(int);

    // Allocate host memory
    A = (int*)malloc(sizeA);
    B = (int*)malloc(sizeB);
    C = (int*)malloc(sizeC);

    // Initialize matrices A and B
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = i + j;
        }
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            B[i * K + j] = i - j;
        }
    }

    // Allocate device memory
    cudaMalloc((void**)&d_A, sizeA);
    cudaMalloc((void**)&d_B, sizeB);
    cudaMalloc((void**)&d_C, sizeC);

    // Transfer data from host to device memory
    cudaMemcpy(d_A, A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeB, cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    clock_t startt, endd;
    startt=clock();
    // Launch the kernel
    matrixMulTiled<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);
endd=clock();
double timee=(double) (endd-startt)/CLOCKS_PER_SEC;
printf("Elapsed time is %f s",timee);
    // Transfer result from device to host memory
    cudaMemcpy(C, d_C, sizeC, cudaMemcpyDeviceToHost);
    

    // Free memory
    free(A);
    free(B);
    free(C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}