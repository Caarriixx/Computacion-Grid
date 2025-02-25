#include <omp.h>
#include <stdio.h>

int main() {
    int nthreads, tid;

    //omp_set_num_threads(8);

    #pragma omp parallel private(tid) num_threads(8)
    {
        // Obtener el ID del hilo actual
        tid = omp_get_thread_num();
        // Obtener el número total de hilos
        nthreads = omp_get_num_threads();

        // Sección paralela con diferentes tareas
        #pragma omp sections
        {
            #pragma omp section
            {
                printf("Thread %d, of %d, computes section 1\n", tid, nthreads);
            }

            #pragma omp section
            {
                printf("Thread %d, of %d, computes section 2\n", tid, nthreads);
            }

            #pragma omp section
            {
                printf("Thread %d, of %d, computes section 3\n", tid, nthreads);
            }

            #pragma omp section
            {
                printf("Thread %d, of %d, computes section 4\n", tid, nthreads);
            }
        } 
    } 

    return 0;
}
