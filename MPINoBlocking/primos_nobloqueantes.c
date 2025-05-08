#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define VECTOR_SIZE 1000000
#define CHUNK_SIZE 1000

int es_primo(int n) {
    if (n < 2) return 0;
    for (int i = 2; i * i <= n; i++) {
        if (n % i == 0) return 0;
    }
    return 1;
}

int main(int argc, char *argv[]) {
    int rank, size;
    double start_time, end_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    start_time = MPI_Wtime();  // Inicia la medición de tiempo

    if (rank == 0) {
        // ======= MAESTRO =======
        int *full_vector = (int*)malloc(VECTOR_SIZE * sizeof(int));
        srand(time(NULL));
        for (int i = 0; i < VECTOR_SIZE; i++) {
            full_vector[i] = rand() % 1000000 + 1;
        }

        int offset = 0;
        int total_primos = 0;
        int activos = size - 1;
        MPI_Status status;

        // Enviar bloques iniciales
        for (int i = 1; i < size && offset < VECTOR_SIZE; i++) {
            MPI_Request request;
            MPI_Isend(&full_vector[offset], CHUNK_SIZE, MPI_INT, i, 0, MPI_COMM_WORLD, &request);
            offset += CHUNK_SIZE;
        }

        // Recibir resultados y seguir enviando
        while (activos > 0) {
            int result, slave_id;

            MPI_Recv(&result, 1, MPI_INT, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &status);
            slave_id = status.MPI_SOURCE;
            total_primos += result;

            if (offset < VECTOR_SIZE) {
                MPI_Request request;
                MPI_Isend(&full_vector[offset], CHUNK_SIZE, MPI_INT, slave_id, 0, MPI_COMM_WORLD, &request);
                offset += CHUNK_SIZE;
            } else {
                int fin[CHUNK_SIZE];
                for (int i = 0; i < CHUNK_SIZE; i++) fin[i] = -1;
                MPI_Request request;
                MPI_Isend(fin, CHUNK_SIZE, MPI_INT, slave_id, 0, MPI_COMM_WORLD, &request);
                activos--;
            }

            printf("Maestro: recibido %d primos del esclavo %d\n", result, slave_id);
        }

        end_time = MPI_Wtime();  // Fin de la medición
        printf("TOTAL DE PRIMOS ENCONTRADOS: %d\n", total_primos);
        printf("Tiempo total con %d procesos: %f segundos\n", size, end_time - start_time);
        free(full_vector);

    } else {
        // ======= ESCLAVOS =======
        int *local_vector = (int*)malloc(CHUNK_SIZE * sizeof(int));
        MPI_Request recv_request, send_request;
        MPI_Status status;

        while (1) {
            MPI_Irecv(local_vector, CHUNK_SIZE, MPI_INT, 0, 0, MPI_COMM_WORLD, &recv_request);
            MPI_Wait(&recv_request, &status);

            if (local_vector[0] == -1) {
                break;
            }

            int count_primos = 0;
            for (int i = 0; i < CHUNK_SIZE; i++) {
                count_primos += es_primo(local_vector[i]);
            }

            MPI_Isend(&count_primos, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &send_request);
        }

        end_time = MPI_Wtime();  // Fin de ejecución del esclavo
        printf("Esclavo %d finalizó en %f segundos\n", rank, end_time - start_time);
        free(local_vector);
    }

    MPI_Finalize();
    return 0;
}
