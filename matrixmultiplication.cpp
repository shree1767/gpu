#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024 // Size of the matrix

__global__ void matrixMul(int *a, int *b, int *c) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;

    if (row < N && col < N) {
        for (int i = 0; i < N; i++) {
            sum += a[row * N + i] * b[i * N + col];
        }
        c[row * N + col] = sum;
    }
}

int main() {
    int *a, *b, *c;           // Host matrices
    int *d_a, *d_b, *d_c;     // Device matrices

    // Allocate memory on the host
    a = (int*)malloc(N * N * sizeof(int));
    b = (int*)malloc(N * N * sizeof(int));
    c = (int*)malloc(N * N * sizeof(int));

    // Initialize matrices a and b
    for (int i = 0; i < N * N; i++) {
        a[i] = i;
        b[i] = i;
    }

    // Allocate memory on the device
    cudaMalloc((void**)&d_a, N * N * sizeof(int));
    cudaMalloc((void**)&d_b, N * N * sizeof(int));
    cudaMalloc((void**)&d_c, N * N * sizeof(int));

    // Copy matrices a and b from host to device
    cudaMemcpy(d_a, a, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * N * sizeof(int), cudaMemcpyHostToDevice);

    // Set the grid and block dimensions
    dim3 gridDim((N + 15) / 16, (N + 15) / 16);
    dim3 blockDim(16, 16);

    // Call the kernel
    matrixMul<<<gridDim, blockDim>>>(d_a, d_b, d_c);

    // Copy matrix c from device to host
    cudaMemcpy(c, d_c, N * N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print matrix c
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", c[i * N + j]);
        }
        printf("\n");
    }

    // Free memory on the device
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Free memory on the host
    free(a);
    free(b);
    free(c);

    return 0;
}
