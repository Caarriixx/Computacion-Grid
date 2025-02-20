#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void sumaparcial(double *a, int inicio, int fin, double *result);
double sumatotal(double *a, int n);
double suma(double *a, int n);

void multparcial(double *a, int inicio, int fin, double *result);
double multtotal(double *a, int n);
double producto(double *a, int n);

int main(){
    int nthreads, tid;
    double vector[100], resultadosuma, resultadoproducto;
    int i, tam = 100;
    srand(time(NULL));
    
    for (i = 0; i < tam; i++) 
        vector[i] = rand() % 5 + 1;
    
    #pragma omp parallel private(tid)
    {
        tid = omp_get_thread_num();
        nthreads = omp_get_num_threads();
        #pragma omp single
        printf("Número total de threads: %d\n", nthreads);
        printf("El thread %d calcula la seccion 1\n", tid);
    }
    
    resultadosuma = suma(vector, tam);
    printf("El resultado de la suma es %f\n", resultadosuma);
    
    #pragma omp parallel private(tid)
    {
        tid = omp_get_thread_num();
        printf("El thread %d calcula la seccion 2\n", tid);
    }
    
    resultadoproducto = producto(vector, tam);
    printf("El resultado del producto es %f\n", resultadoproducto);
    
    return 0;
}

void sumaparcial(double *a, int inicio, int fin, double *result) {
    int i;
    double s = 0;
    for (i = inicio; i < fin; i++)
        s += a[i];
    *result = s;
}

double sumatotal(double *a, int n) {
    double total = 0;
    #pragma omp parallel for reduction(+:total)
    for (int i = 0; i < n; i++)
        total += a[i];
    return total;
}

double suma(double *a, int n) {
    return sumatotal(a, n);
}

void multparcial(double *a, int inicio, int fin, double *result) {
    int i;
    double p = 1;
    for (i = inicio; i < fin; i++)
        p *= a[i];
    *result = p;
}

double multtotal(double *a, int n) {
    double total = 1;
    #pragma omp parallel for reduction(*:total)
    for (int i = 0; i < n; i++)
        total *= a[i];
    return total;
}

double producto(double *a, int n) {
    return multtotal(a, n);
}