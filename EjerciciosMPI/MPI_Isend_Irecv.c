#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

int main(int argc, char *argv[]) {
    int rank, size;
    char hostname[MPI_MAX_PROCESSOR_NAME];
    char recv_buffer[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Request reqs[2];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Get_processor_name(hostname, &name_len);

    int next = (rank + 1) % size;
    int prev = (rank - 1 + size) % size;

    MPI_Irecv(recv_buffer, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, prev, 0, MPI_COMM_WORLD, &reqs[0]);
    MPI_Isend(hostname, name_len + 1, MPI_CHAR, next, 0, MPI_COMM_WORLD, &reqs[1]);

    MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);

    printf("Proceso %d recibió de %d: %s\n", rank, prev, recv_buffer);
    MPI_Finalize();
    return 0;
}


/* 
¿Pueden ejecutarse en cualquier orden Send()/Isend() y Recv()/Irecv()?

No.
MPI_Recv() bloquea hasta que llega el mensaje.

MPI_Send() puede bloquear o no dependiendo del sistema, pero puede provocar interbloqueos si no se sincronizan bien los procesos.

Por eso, no se pueden ejecutar en cualquier orden, especialmente con Send() y Recv().
En el caso de MPI_Isend() y MPI_Irecv(), aunque no bloquean, deben acompañarse de MPI_Wait o MPI_Test, y el orden sigue siendo importante para evitar condiciones de carrera o pérdida de mensajes.
*/