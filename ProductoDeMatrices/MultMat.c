#include <stdio.h>

void multiplyMatrices(int firstMatrix[][10], int secondMatrix[][10], int result[][10], int row1, int col1, int col2) {
    for (int i = 0; i < row1; i++) {
        for (int j = 0; j < col2; j++) {
            result[i][j] = 0;
            for (int k = 0; k < col1; k++) {
                result[i][j] += firstMatrix[i][k] * secondMatrix[k][j];
            }
        }
    }
}

void inputMatrix(int matrix[][10], int row, int col) {
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            printf("Elemento [%d][%d]: ", i + 1, j + 1);
            scanf("%d", &matrix[i][j]);
        }
    }
}

void printMatrix(int matrix[][10], int row, int col) {
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            printf("%d ", matrix[i][j]);
        }
        printf("\n");
    }
}

int main() {
    int row1, col1, row2, col2;
    int firstMatrix[10][10], secondMatrix[10][10], result[10][10];

    printf("Ingrese filas y columnas de la primera matriz: ");
    scanf("%d %d", &row1, &col1);

    printf("Ingrese filas y columnas de la segunda matriz: ");
    scanf("%d %d", &row2, &col2);

    if (col1 != row2) {
        printf("Error: No se pueden multiplicar estas matrices.\n");
        return 1;
    }

    printf("Ingrese los elementos de la primera matriz:\n");
    inputMatrix(firstMatrix, row1, col1);

    printf("Ingrese los elementos de la segunda matriz:\n");
    inputMatrix(secondMatrix, row2, col2);

    multiplyMatrices(firstMatrix, secondMatrix, result, row1, col1, col2);

    printf("Resultado de la multiplicación:\n");
    printMatrix(result, row1, col2);

    return 0;
}
