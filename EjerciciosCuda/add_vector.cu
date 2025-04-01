#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

// Kernel para inicializar el vector con su índice
__global__ void init_vector(int *vec, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < n; i += stride) {
        vec[i] = i;
    }
}

// Kernel para sumar todos los elementos del vector
__global__ void add_vector(int *vec, int n, int *result) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int local_sum = 0;
    for (int i = tid; i < n; i += stride) {
        local_sum += vec[i];
    }
    atomicAdd(result, local_sum);
}

int main(int argc, char *argv[]) {
    int n = 32;   // tamaño del vector
    int th = 8;   // hilos por bloque
    int bl = 4;   // bloques

    if (argc > 1) n = atoi(argv[1]);
    if (argc > 2) th = atoi(argv[2]);
    if (argc > 3) bl = atoi(argv[3]);

    int *h_vec = (int *)malloc(n * sizeof(int));
    int *d_vec, *d_result;
    int h_result = 0;

    cudaMalloc((void**)&d_vec, n * sizeof(int));
    cudaMalloc((void**)&d_result, sizeof(int));
    cudaMemset(d_result, 0, sizeof(int));

    // Inicializar vector en GPU
    init_vector<<<bl, th>>>(d_vec, n);
    cudaDeviceSynchronize();

    // Sumar en GPU
    add_vector<<<bl, th>>>(d_vec, n, d_result);
    cudaDeviceSynchronize();

    // Copiar resultados
    cudaMemcpy(h_vec, d_vec, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

    // Imprimir vector y suma
    printf("Vector:\n");
    for (int i = 0; i < n; i++) {
        printf("%d ", h_vec[i]);
    }
    printf("\nSuma total: %d\n", h_result);

    // Liberar memoria
    free(h_vec);
    cudaFree(d_vec);
    cudaFree(d_result);

    return 0;
}
