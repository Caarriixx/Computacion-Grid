
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <omp.h>

#define N 512
#define TILE_WIDTH 16

__global__ void matrixMulShared(float* A, float* B, float* C, int width) {
    // Allocate shared memory for the two blocks aSub and bSub.
    __shared__ float aSub[TILE_WIDTH][TILE_WIDTH];
    __shared__ float bSub[TILE_WIDTH][TILE_WIDTH];

    // Calculate global thread index 
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Calculate global offset of upper left corner of thread block.
    int Row = blockIdx.y * TILE_WIDTH + ty;
    int Col = blockIdx.x * TILE_WIDTH + tx;

    float sum = 0.0f;
    for (int t = 0; t < width / TILE_WIDTH; ++t) {
        // Copy block into shared memory
        aSub[ty][tx] = A[Row * width + (t * TILE_WIDTH + tx)];
        bSub[ty][tx] = B[(t * TILE_WIDTH + ty) * width + Col];

        // Synchronize threads to make sure all threads are done copying
        __syncthreads();

        for (int i = 0; i < TILE_WIDTH; ++i)
            sum += aSub[ty][i] * bSub[i][tx];

        // Synchronize threads to make sure all threads are done with the data
        __syncthreads();
    }

    if (Row < width && Col < width)
        C[Row * width + Col] = sum;
}

int main() {
    int size = N * N * sizeof(float);
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;

    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);

    for (int i = 0; i < N * N; ++i) {
        h_A[i] = 1.0f;
        h_B[i] = 1.0f;
    }

    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid(N / TILE_WIDTH, N / TILE_WIDTH);

    // Call the kernel 
    matrixMulShared<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    printf("Sample result C[0]: %f\n", h_C[0]);

    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    return 0;
}
