#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

int main(int argc, char *argv[]) {
    int rank, size;
    char hostname[MPI_MAX_PROCESSOR_NAME];
    char recv_buffer[MPI_MAX_PROCESSOR_NAME];
    int name_len;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Get_processor_name(hostname, &name_len);

    int next = (rank + 1) % size;
    int prev = (rank - 1 + size) % size;

    if (rank == 0) {
        MPI_Send(hostname, name_len + 1, MPI_CHAR, next, 0, MPI_COMM_WORLD);
        MPI_Recv(recv_buffer, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, prev, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else {
        MPI_Recv(recv_buffer, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, prev, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(hostname, name_len + 1, MPI_CHAR, next, 0, MPI_COMM_WORLD);
    }

    printf("Proceso %d recibió de %d: %s\n", rank, prev, recv_buffer);
    MPI_Finalize();
    return 0;
}
