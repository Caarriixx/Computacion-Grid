#include <stdio.h>
#include <stdlib.h>

// Kernel que inicializa cada posición con su índice
__global__ void init_vector(int *vec, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n) {
        vec[index] = index;
    }
}

int main(int argc, char *argv[]) {
    int n = 32; // tamaño por defecto
    int th = 8; // threads por bloque por defecto

    // Leer argumentos si los hay
    if (argc > 1) n = atoi(argv[1]);
    if (argc > 2) th = atoi(argv[2]);

    if (n % th != 0) {
        printf("Error: el número de threads por bloque debe ser múltiplo de n.\n");
        return 1;
    }

    int *h_vec, *d_vec;
    size_t size = n * sizeof(int);

    // Reservar memoria en host y device
    h_vec = (int *)malloc(size);
    cudaMalloc((void **)&d_vec, size);

    int blocks = n / th;

    // Lanzar kernel
    init_vector<<<blocks, th>>>(d_vec, n);

    // Copiar de device a host
    cudaMemcpy(h_vec, d_vec, size, cudaMemcpyDeviceToHost);

    // Imprimir el vector
    printf("Vector resultante:\n");
    for (int i = 0; i < n; i++) {
        printf("%d ", h_vec[i]);
    }
    printf("\n");

    // Liberar memoria
    cudaFree(d_vec);
    free(h_vec);

    return 0;
}
