#include <stdio.h>
#include <time.h>

#define M 10000
#define N 10000
#define K 10000

__global__ void matrixMul(int *A, int *B, int *C, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < k) {
        int sum = 0;
        for (int i = 0; i < n; i++) {
            sum += A[row * n + i] * B[i * k + col];
        }
        C[row * k + col] = sum;
    }
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
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start event
    clock_t startt, endd;
    startt=clock();

    // Launch the kernel
    matrixMul<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);

    endd=clock();
    double seconds = (double) (endd-startt)/CLOCKS_PER_SEC;
    
    printf("Elapsed Time: %f s\n", seconds);

    // Transfer result from device to host memory
    cudaMemcpy(C, d_C, sizeC, cudaMemcpyDeviceToHost);

    // Print the result
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            //printf("%4d ", C[i * K + j]);
        }
        //printf("\n");
    }

    // Free memory
    free(A);
    free(B);
    free(C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
