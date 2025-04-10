#include <stdio.h>
#include <mpi.h>

#define N 12  // Tamaño total del vector (debe ser divisible entre el número de procesos)

int main(int argc, char *argv[]) {
    int pid, npr, i;
    int VA[N];         // Vector original solo en el maestro
    int local_size;    // Tamaño del subvector por proceso
    int local_VA[N];   // Buffer local para cada proceso
    int suma_local = 0, suma_total = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &npr);

    local_size = N / npr;

    if (pid == 0) {
        for (i = 0; i < N; i++) {
            VA[i] = i + 1;  // Por ejemplo: 1, 2, ..., 12
        }
    }

    MPI_Scatter(VA, local_size, MPI_INT, local_VA, local_size, MPI_INT, 0, MPI_COMM_WORLD);

    for (i = 0; i < local_size; i++) {
        local_VA[i] *= 2;
        suma_local += local_VA[i];
    }

    MPI_Reduce(&suma_local, &suma_total, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (pid == 0) {
        printf("El maestro recibió la suma total de los valores multiplicados por 2: %d\n", suma_total);
    }

    MPI_Finalize();
    return 0;
}
