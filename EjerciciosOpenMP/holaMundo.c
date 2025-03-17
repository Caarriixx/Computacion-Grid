#include <stdio.h>
#include <omp.h>

int main() {
    int nthreads, tid;

    printf("Set 4 threads\n");

    // Set number of threads to 4
    omp_set_num_threads(4);

    // Start parallel region
    #pragma omp parallel private(tid)
    {
        // Get the ID of the thread
        tid = omp_get_thread_num();
        printf("Hello from thread = %d\n", tid);

        // If I am the master thread
        if (tid == 0) {
            nthreads = omp_get_num_threads();
            printf("Number of threads = %d\n", nthreads);
        }
    }

    printf("Set 3 threads\n");

    // Set number of threads to 3
    omp_set_num_threads(3);

    // Start second parallel region
    #pragma omp parallel private(tid)
    {
        // Get the ID of the thread
        tid = omp_get_thread_num();
        printf("Hello from thread = %d\n", tid);

        // If I am the master thread
        if (tid == 0) {
            nthreads = omp_get_num_threads();
            printf("Number of threads = %d\n", nthreads);
        }
    }

    return 0;
}

