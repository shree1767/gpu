#include <stdio.h>
__global void helloCUDA()
{
printf("Hello CUDA World!");
}

int main() {
  helloCUDA<<<1, 1>>>(); 
  cudaDeviceSynchronize(); 
  return 0;
}
