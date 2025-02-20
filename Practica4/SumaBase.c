#include <omp.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>

void sumaparcial(double *a, int n);
double sumatotal(double *a, int n, int THREADS);
double suma(double *a, int n, int THREADS);

int main() {
    double vector[100], resultado;
    int i, tam = 100;

    srand(time(NULL)); // Semilla aleatoria
    for (i = 0; i < tam; i++) 
        vector[i] = rand() % 100; // Valores aleatorios entre 0 y 99

    resultado = suma(vector, tam, 8);
    printf("El resultado es %f\n", resultado);
    return 0;
}

// Función que suma los n primeros elementos de un subvector y deja el resultado en a[0]
void sumaparcial(double *a, int n) {
    int i;
    double suma_local = 0.0;

    for (i = 0; i < n; i++) {
        suma_local += a[i];
    }

    a[0] = suma_local; // Guardar la suma en la primera posición del subvector
}

// Función que calcula la suma total sumando las sumas parciales de cada subvector
double sumatotal(double *a, int n, int THREADS) {
    double total = 0.0;
    int i;

    // Sumar los valores de la primera posición de cada subvector
    #pragma omp parallel for reduction(+:total)
    for (i = 0; i < THREADS; i++) {
        total += a[i * n / THREADS];
    }

    return total;
}

// Función que reparte la suma en hilos
double suma(double *a, int n, int THREADS) {
    int i;

    // Paralelizar el cálculo de sumas parciales
    #pragma omp parallel for
    for (i = 0; i < THREADS; i++) {
        sumaparcial(&a[i * n / THREADS], n / THREADS);
    }

    // Sumar los resultados parciales y devolver el total
    return sumatotal(a, n, THREADS);
}
