#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>    // Para números aleatorios

int main(int argc, char *argv[]) {
    int size, rank;
    int N;
    int **A = NULL;
    int **B = NULL;
    int **C = NULL;
    int *row_A = NULL;
    int *row_B = NULL;
    int *row_C = NULL;
    double start_time, end_time;  // Para medir el tiempo

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    N = size;  // El tamaño de las matrices es el número de procesos

    // Reservar espacio para la fila de cada proceso
    row_A = (int *)malloc(N * sizeof(int));
    row_B = (int *)malloc(N * sizeof(int));
    row_C = (int *)malloc(N * sizeof(int));  // Aquí guardaremos el resultado de la multiplicación

    if (rank == 0) {
        // Reservamos memoria para las matrices
        A = (int **)malloc(N * sizeof(int *));
        B = (int **)malloc(N * sizeof(int *));
        C = (int **)malloc(N * sizeof(int *));
        for (int i = 0; i < N; i++) {
            A[i] = (int *)malloc(N * sizeof(int));
            B[i] = (int *)malloc(N * sizeof(int));
            C[i] = (int *)malloc(N * sizeof(int));
        }

        // Inicializamos los valores de las matrices con números aleatorios entre 0 y 100
        srand(time(NULL));  // Semilla de números aleatorios
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                A[i][j] = rand() % 101;
                B[i][j] = rand() % 101;
            }
        }

        // MEDIR TIEMPO: justo antes de empezar a distribuir
        start_time = MPI_Wtime();

        // DISTRIBUCIÓN de las filas
        for (int dest = 1; dest < size; dest++) {
            MPI_Send(A[dest], N, MPI_INT, dest, 0, MPI_COMM_WORLD);
            MPI_Send(B[dest], N, MPI_INT, dest, 0, MPI_COMM_WORLD);
        }

        // El maestro copia su propia fila
        for (int j = 0; j < N; j++) {
            row_A[j] = A[0][j];
            row_B[j] = B[0][j];
        }
    } else {
        // Los esclavos reciben su fila de A y su fila de B
        MPI_Recv(row_A, N, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(row_B, N, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Cada proceso multiplica su fila
    for (int i = 0; i < N; i++) {
        row_C[i] = row_A[i] * row_B[i];
    }

    // Ahora cada proceso envía su fila C al maestro
    if (rank == 0) {
        // El maestro copia su propia fila en la matriz C
        for (int j = 0; j < N; j++) {
            C[0][j] = row_C[j];
        }

        // El maestro recibe el resto de las filas
        for (int source = 1; source < size; source++) {
            MPI_Recv(C[source], N, MPI_INT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // MEDIR TIEMPO: justo después de reconstruir la matriz C
        end_time = MPI_Wtime();

        printf("Tiempo de ejecución: %f segundos\n", end_time - start_time);
    } else {
        // Los esclavos envían su fila
        MPI_Send(row_C, N, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
