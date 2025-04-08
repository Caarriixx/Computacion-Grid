#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int pid, npr;
    int VA[10], i;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &npr);

    // Solo el proceso 0 inicializa el vector
    if (pid == 0) {
        for (i = 0; i < 10; i++) {
            VA[i] = i;
        }
    }

    // Todos los procesos (incluido el 0) hacen el broadcast
    MPI_Bcast(VA, 10, MPI_INT, 0, MPI_COMM_WORLD);

    // Todos imprimen el vector recibido
    printf("Proceso %d recibió el vector:\n", pid);
    for (i = 0; i < 10; i++) {
        printf("%d ", VA[i]);
    }
    printf("\n");

    MPI_Finalize();
    return 0;
}
