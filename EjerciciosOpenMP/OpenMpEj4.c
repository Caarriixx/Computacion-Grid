#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 4
#define M 4

int main() {
    int i, j, nthreads, tid, n, m, sum, a[M], c[N], b[M][N];

    srand(time(NULL)); // Inicializar la semilla para números aleatorios
    m = M;
    n = N;

    // Inicializar la matriz b con valores aleatorios
    for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++) {
            b[i][j] = rand() % 100; 
        }
    }

    // Inicializar el vector c con valores aleatorios
    for (i = 0; i < N; i++) {
        c[i] = rand() % 100;
    }

    // Configurar el número de hilos a 8
    omp_set_num_threads(8);

    // Paralelizar el cálculo de a[i]
    #pragma omp parallel for private(i, j, sum, tid) shared(a, b, c, m, n, nthreads)
    for (i = 0; i < m; i++) {
        tid = omp_get_thread_num();
        nthreads = omp_get_num_threads();
        sum = 0;

        for (j = 0; j < n; j++) {
            sum += b[i][j] * c[j];
        }

        a[i] = sum;

        printf("Thread %d, of %d threads, computes iteration i=%d\n", tid, nthreads, i);
    }

    // Imprimir el resultado del vector a
    for (i = 0; i < M; i++) {
        printf("a[%d]=%d\n", i, a[i]);
    }

    return 0;
}
