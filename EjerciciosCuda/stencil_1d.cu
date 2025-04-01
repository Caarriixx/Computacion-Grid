#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>

#define N 1024  // tamaño del vector
#define RADIUS 1  // radio del stencil

__global__ void stencil_1d_not_shared(int *in, int *out, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= RADIUS && i < n - RADIUS) {
        out[i] = (in[i - 1] + in[i] + in[i + 1]) / 3;
    }
}

__global__ void stencil_1d_shared(int *in, int *out, int n) {
    extern __shared__ int temp[];
    int tid = threadIdx.x;
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    // cargar a memoria compartida
    if (i < n) {
        temp[tid + RADIUS] = in[i];
        if (tid == 0 && i > 0)
            temp[tid] = in[i - 1];
        if (tid == blockDim.x - 1 && i < n - 1)
            temp[tid + 2 * RADIUS] = in[i + 1];
    }

    __syncthreads();

    if (i >= RADIUS && i < n - RADIUS) {
        out[i] = (temp[tid] + temp[tid + 1] + temp[tid + 2]) / 3;
    }
}

void init_vector(int *v, int n) {
    for (int i = 0; i < n; i++) {
        v[i] = rand() % 100;
    }
}

void print_vector(const char *name, int *v, int n) {
    printf("%s:\n", name);
    for (int i = 0; i < n; i++) {
        printf("%d ", v[i]);
    }
    printf("\n");
}

int main(int argc, char *argv[]) {
    int blockSize = 256;  // valor por defecto

    if (argc > 1) {
        blockSize = atoi(argv[1]);
    }

    int *in, *out1, *out2;
    int *d_in, *d_out1, *d_out2;

    size_t size = N * sizeof(int);
    in = (int *)malloc(size);
    out1 = (int *)malloc(size);
    out2 = (int *)malloc(size);

    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out1, size);
    cudaMalloc(&d_out2, size);

    srand(time(NULL));
    init_vector(in, N);

    cudaMemcpy(d_in, in, size, cudaMemcpyHostToDevice);

    int numBlocks = (N + blockSize - 1) / blockSize;

    // tiempo no_shared
    cudaEvent_t start1, stop1;
    cudaEventCreate(&start1); cudaEventCreate(&stop1);
    cudaEventRecord(start1);
    stencil_1d_not_shared<<<numBlocks, blockSize>>>(d_in, d_out1, N);
    cudaEventRecord(stop1);
    cudaEventSynchronize(stop1);
    float time1;
    cudaEventElapsedTime(&time1, start1, stop1);

    // tiempo shared
    cudaEvent_t start2, stop2;
    cudaEventCreate(&start2); cudaEventCreate(&stop2);
    cudaEventRecord(start2);
    stencil_1d_shared<<<numBlocks, blockSize, (blockSize + 2 * RADIUS) * sizeof(int)>>>(d_in, d_out2, N);
    cudaEventRecord(stop2);
    cudaEventSynchronize(stop2);
    float time2;
    cudaEventElapsedTime(&time2, start2, stop2);

    cudaMemcpy(out1, d_out1, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(out2, d_out2, size, cudaMemcpyDeviceToHost);

    // Imprimir resultados
    print_vector("Input", in, 10);
    print_vector("Stencil not shared", out1, 10);
    print_vector("Stencil shared", out2, 10);

    printf("Tiempo sin shared memory: %.3f ms\n", time1);
    printf("Tiempo con shared memory: %.3f ms\n", time2);

    free(in); free(out1); free(out2);
    cudaFree(d_in); cudaFree(d_out1); cudaFree(d_out2);

    return 0;
}
