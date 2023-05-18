#include <stdio.h>
#include <stdlib.h>
#include <time.h>

 int m=10000;
 int n=10000;
 int p=10000;
// Function to generate random values for a matrix
void generateRandomMatrix(int rows, int cols, int matrix[rows][cols]) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i][j] = rand() % 10;  // Generate random numbers from 0 to 9
        }
    }
}

// Function to multiply two matrices and store the result in another matrix
void matrixMultiplication(int m, int n, int p, int matrix1[m][n], int matrix2[n][p], int result[m][p]) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            result[i][j] = 0;
            for (int k = 0; k < n; k++) {
                result[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }
}

int main() {

    // Check if matrix multiplication is possible
    if (n <= 0 || m <= 0 || p <= 0) {
        printf("Invalid dimensions! Exiting the program.\n");
        return 1;
    }

    int matrix1[m][n], matrix2[n][p], result[m][p];

    // Generate random matrices
    generateRandomMatrix(m, n, matrix1);
    generateRandomMatrix(n, p, matrix2);

    clock_t startTime = clock();

    // Multiply the matrices
    matrixMultiplication(m, n, p, matrix1, matrix2, result);

    clock_t endTime = clock();

    double executionTime = (double)(endTime - startTime) / CLOCKS_PER_SEC;

    // Print the result matrix
    printf("Result Matrix:\n");
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            printf("%d ", result[i][j]);
        }
        printf("\n");
    }

    printf("Execution Time: %.6f seconds\n", executionTime);

    return 0;
}
