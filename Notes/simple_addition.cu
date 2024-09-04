#include <stdio.h>
#include <cuda.h>

// CUDA kernel for vector addition on GPU
__global__
void vecAddKernel(float *A_d, float *B_d, float *C_d, int N)
{
    // Calculate the global thread index
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Add if index is within index bounds
    if(i < N) 
        C_d[i] = A_d[i] + B_d[i];
}

// Host function to set up and launch the CUDA kernel
__host__
void vecAdd(float *A_d, float *B_d, float *C_d, int N)
{
    // Grid dimension
    dim3 DimGrid(ceil(N/256.0), 1, 1);

    // Block dimension
    dim3 DimBlock(256, 1, 1);

    // Launch the kernel
    vecAddKernel<<<DimGrid, DimBlock>>>(A_d, B_d, C_d, N);
} 

int main()
{
    // Number of elements in a vector
    int N = 1000;
    
    // Host vectors
    float *A_h, *B_h, *C_h;
    
    // Device vectors
    float *A_d, *B_d, *C_d;
    
    // Size in bytes for N elements
    size_t size = N * sizeof(float);
    
    // Allocate memory on host
    A_h = (float*)malloc(size);
    B_h = (float*)malloc(size);
    C_h = (float*)malloc(size);
    
    // Initialize host vectors
    for(int i = 0; i < N; i++) {
        A_h[i] = i;
        B_h[i] = i * 2;
    }
    
    // Allocate memory on device
    cudaMalloc((void**)&A_d, size);
    cudaMalloc((void**)&B_d, size);
    cudaMalloc((void**)&C_d, size);
    
    // Copy data from host to device
    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);
    
    // Perform vector addition
    vecAdd(A_d, B_d, C_d, N);
    
    // Copy result from device to host
    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);
    
    // Print the result
    for(int i = 0; i < 10; i++) {
        printf("%f + %f = %f\n", A_h[i], B_h[i], C_h[i]);
    }
    
    // Free memory
    free(A_h);
    free(B_h);
    free(C_h);
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
    
    return 0;
}