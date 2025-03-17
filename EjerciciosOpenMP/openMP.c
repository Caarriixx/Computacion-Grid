#include <omp.h>
#include <stdio.h>

#define CHUNKSIZE 2
#define N 10

int main() {
    int i, chunk, nthreads, tid;
    int a[N], b[N], c[N];

    for (i = 0; i < N; i++)
        a[i] = b[i] = i * 1.0;

    chunk = CHUNKSIZE;

    // Establecer el número de hilos en 8
    omp_set_num_threads(8);

    #pragma omp parallel private(i, tid) shared(a, b, c, chunk, nthreads)
    {
        // Obtener el ID del hilo
        tid = omp_get_thread_num();
        
        // Obtener el número total de hilos solo una vez
        #pragma omp single
        nthreads = omp_get_num_threads();

        // Bucle paralelo con reparto estático en bloques de tamaño "chunk"
        #pragma omp for schedule(static, chunk) nowait
        for (i = 0; i < N; i++) {
            c[i] = a[i] + b[i];
            printf("Thread %d, of %d threads, computes iteration i = %d\n", tid, nthreads, i);
        }

        printf("Thread %d ends\n", tid);
    } 

    printf("Parallel region ends");

    return 0;
}

