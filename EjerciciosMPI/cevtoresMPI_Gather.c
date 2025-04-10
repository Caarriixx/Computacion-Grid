#include <stdio.h>
#include <mpi.h>

#define N 12  // Tamaño total del vector (divisible por número de procesos)

int main(int argc, char *argv[]) {
    int pid, npr, i;
    int VA[N];         // Vector original en el maestro
    int local_size;    // Tamaño del subvector que recibe cada proceso
    int local_VA[N];   // Buffer local para cada proceso
    int result[N];     // Vector para recoger el resultado final en el maestro

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &npr);

    local_size = N / npr;

    if (pid == 0) {
        for (i = 0; i < N; i++) {
            VA[i] = i + 1;  // por ejemplo [1, 2, ..., 12]
        }
    }

    MPI_Scatter(VA, local_size, MPI_INT, local_VA, local_size, MPI_INT, 0, MPI_COMM_WORLD);

    for (i = 0; i < local_size; i++) {
        local_VA[i] *= 2;  // Multiplicamos cada elemento por 2
    }

    MPI_Gather(local_VA, local_size, MPI_INT, result, local_size, MPI_INT, 0, MPI_COMM_WORLD);

    if (pid == 0) {
        printf("El maestro reunió el vector final:\n");
        for (i = 0; i < N; i++) {
            printf("%d ", result[i]);
        }
        printf("\n");
    }

    MPI_Finalize();
    return 0;
}
