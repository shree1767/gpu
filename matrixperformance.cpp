#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define TILE_WIDTH 32

__global__ void matrix_multiply(float *a, float *b, float *c, int n) {
    __shared__ float ds_a[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_b[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;
    float sum = 0.0;

    for (int i = 0; i < n / TILE_WIDTH; i++) {
        ds_a[ty][tx] = a[row * n + i * TILE_WIDTH + tx];
        ds_b[ty][tx] = b[(i * TILE_WIDTH + ty) * n + col];

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; k++) {
            sum += ds_a[ty][k] * ds_b[k][tx];
        }

        __syncthreads();
    }

    c[row * n + col] = sum;
}

int main() {
    int n = 1024;
    float *a, *b, *c;
    float *dev_a, *dev_b, *dev_c;

    a = (float*)malloc(n * n * sizeof(float));
    b = (float*)malloc(n * n * sizeof(float));
    c = (float*)malloc(n * n * sizeof(float));

    cudaMalloc((void**)&dev_a, n * n * sizeof(float));
    cudaMalloc((void**)&dev_b, n * n * sizeof(float));
    cudaMalloc((void**)&dev_c, n * n * sizeof(float));

    for (int i = 0; i < n * n; i++) {
        a[i] = 1.0;
        b[i] = 2.0;
        c[i] = 0.0;
    }

    cudaMemcpy(dev_a, a, n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, n * n * sizeof(float), cudaMemcpyHostToDevice);

    dim3 dimGrid(n / TILE_WIDTH, n / TILE_WIDTH, 1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    matrix_multiply<<<dimGrid, dimBlock>>>(dev_a, dev_b, dev_c, n);

    cudaMemcpy(c, dev_c, n * n * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n * n; i++) {
        if (c[i] != n * 2) {
            printf("Error: matrix multiplication failed\n");
            break;
        }
    }

    printf("Matrix multiplication successful\n");

    free(a);
    free(b);
    free(c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}
