// Librería que contiene las funciones scanf y printf
#include <stdio.h> 
#include <time.h>
#include <stdlib.h>
#include <sys/time.h>

// Definimos las constantes del programa
#define ORDEN 100

double wallclock(void);

// Función principal del programa
int main (int argc, char *argv[]) 
{
    // Declaración de variables
    int i, j, t, s, n;
    int mA[ORDEN][ORDEN], mB[ORDEN][ORDEN], mC[ORDEN][ORDEN];	
    double time_cpu, t0, t1;
    
    n = ORDEN;
    srand(time(NULL));

    // Inicialización de matrices con valores aleatorios
    for (i = 0; i < ORDEN; i++) {
        for (j = 0; j < ORDEN; j++) {
            mA[i][j] = rand() % 100;
            mB[i][j] = rand() % 100;
        }
    }

    // Medición de tiempo antes de la ejecución
    t0 = wallclock();

    // Paralelización de la multiplicación de matrices
    #pragma omp parallel for private(j, t, s) shared(mA, mB, mC, n)
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            s = 0;
            for (t = 0; t < n; t++)
                s += mA[i][t] * mB[t][j];
            mC[i][j] = s;
        }
    }

    // Medición de tiempo después de la ejecución
    t1 = wallclock();
    time_cpu = (t1 - t0);

    // Imprimir la matriz resultado
    printf("\n");
    for (i = 0; i < ORDEN; i++) {
        for (j = 0; j < ORDEN; j++) {
            printf("%d\t", mC[i][j]);
        }
        printf("\n");
    }

    // Imprimir el tiempo de ejecución en milisegundos
    printf("\nSe ha tardado %f ms\n", time_cpu * 1e3);
    
    return 0;
}

// Función para medir el tiempo de ejecución
double wallclock(void)
{
    struct timeval tv;
    double t;

    gettimeofday(&tv, NULL);

    t = (double)tv.tv_sec;
    t += ((double)tv.tv_usec) / 1000000.0;

    return t;
}
