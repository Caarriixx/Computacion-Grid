
#include <stdio.h>
#include <mpi.h>

int main(int argc, char* argv[]){

	int size, rank, i, len;
	char message[100], name[MPI_MAX_PROCESSOR_NAME];
	MPI_Status status;
	
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD,&size);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Get_processor_name(name,&len);
	
	if (rank == 0){
		for (i=1; i<size; i++) {
			MPI_Recv(message,100,MPI_CHAR,i,0,MPI_COMM_WORLD,&status);
			printf(message);
		}
	}else {
		sprintf(message,"hola mundo paralelo, soy el proceso %d de %d en el nodo %s\n",rank,size,name);
		MPI_Send(message,100,MPI_CHAR,0,0,MPI_COMM_WORLD);
	}

	MPI_Finalize();
	return 0;
}