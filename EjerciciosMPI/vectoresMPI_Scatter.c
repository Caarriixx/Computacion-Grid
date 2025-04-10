#include <stdio.h>
#include <mpi.h>

#define N 12  // tamaño del vector total, debe ser divisible por el número de procesos

int main(int argc, char *argv[]) {
    int pid, npr, i;
    int VA[N];  // vector completo solo lo tendrá el proceso 0
    int local_size;  // tamaño del subvector para cada proceso
    int local_VA[N]; // espacio máximo, luego usaremos solo local_size

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &npr);

    local_size = N / npr;

    if (pid == 0) {
        for (i = 0; i < N; i++) {
            VA[i] = i;
        }
    }

    MPI_Scatter(VA, local_size, MPI_INT, local_VA, local_size, MPI_INT, 0, MPI_COMM_WORLD);

    printf("Proceso %d recibió: ", pid);
    for (i = 0; i < local_size; i++) {
        printf("%d ", local_VA[i]);
    }
    printf("\n");

    MPI_Finalize();
    return 0;
}
