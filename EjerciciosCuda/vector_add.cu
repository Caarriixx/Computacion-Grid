#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 512  // tamaño del vector

__global__ void add(int *a, int *b, int *c) {
    int index = threadIdx.x;
    c[index] = a[index] + b[index];
}

int main() {
    srand(time(NULL));

    int *a, *b, *c;             // copias en host
    int *d_a, *d_b, *d_c;       // copias en device
    int size = N * sizeof(int);

    // Reservar memoria en host
    a = (int *)malloc(size);
    b = (int *)malloc(size);
    c = (int *)malloc(size);

    // Inicializar vectores a y b con valores aleatorios
    for (int i = 0; i < N; ++i) {
        a[i] = rand() % 11 + 5;
        b[i] = rand() % 11 + 5;
    }

    // Reservar memoria en device
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Copiar a y b a device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // Lanzar kernel (N hilos en 1 bloque)
    add<<<1, N>>>(d_a, d_b, d_c);

    // Copiar resultado de c de device a host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // Mostrar los primeros 10 resultados para comprobar
    printf("a[i] + b[i] = c[i]:\n");
    for (int i = 0; i < 10; i++) {
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }

    // Liberar memoria
    free(a); free(b); free(c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    return 0;
}
