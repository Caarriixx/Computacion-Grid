#include <omp.h>
#include <stdio.h>
#include <math.h>

double funcion(double a);

int main(int argc, char *argv[]){
    int n, i;
    double PI25DT = 3.141592653589793238462643; 
    double h, sum, x;
    
    n = 100; // Número de intervalos
    h = 1.0 / (double)n;
    sum = 0.0; // Suma de intervalos

    // Paralelizar el cálculo de sum con reducción para evitar condiciones de carrera
    #pragma omp parallel for reduction(+:sum) private(i, x)
    for (i = 1; i <= n; i++) {
        x = h * ((double)i - 0.5);
        sum += funcion(x);
    }

    sum = h * sum; // Multiplicar por h después de la reducción

    printf("\nPi es aproximadamente %.16f. El error cometido es %.16f\n", sum, fabs(sum - PI25DT));
    return 0;
}

double funcion(double a) {
    return (4.0 / (1.0 + a * a));
}
