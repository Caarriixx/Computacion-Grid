#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int pid, npr, origen, destino, ndat, tag = 0, esc;
    int VA[10], i, j, num, total = 10;
    MPI_Status info;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &npr);

    ndat = total / (npr - 1); // npr incluye al maestro
    int resto = total % (npr - 1);

    if (pid == 0) {
        // Inicializar el vector VA con datos del 0 al 9
        for (i = 0; i < total; i++)
            VA[i] = i;

        // Enviar partes del vector a los esclavos
        for (destino = 1; destino < npr; destino++) {
            int cantidad = (destino == npr - 1) ? ndat + resto : ndat;
            MPI_Send(&VA[(destino - 1) * ndat], cantidad, MPI_INT, destino, tag, MPI_COMM_WORLD);
        }

    } else {
        // Parte del esclavo
        int local[10] = {0}; // inicializar en 0 para mostrar antes de recibir
        printf("Proceso %d: VA antes de recibir datos:\n", pid);
        for (i = 0; i < total; i++) printf("%d ", local[i]);
        printf("\n");

        int cantidad = (pid == npr - 1) ? ndat + resto : ndat;
        MPI_Recv(local, cantidad, MPI_INT, 0, tag, MPI_COMM_WORLD, &info);

        printf("Proceso %d recibe VA de %d: tag %d, ndat %d;\nVA = ", pid, info.MPI_SOURCE, info.MPI_TAG, cantidad);
        for (i = 0; i < cantidad; i++) printf("%d ", local[i]);
        printf("\n\n");
    }

    MPI_Finalize();
    return 0;
}
