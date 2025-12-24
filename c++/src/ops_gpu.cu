#include "../include/ops_gpu.h"
#include <cuda_runtime.h>
#include <stdexcept>

// MatMul Kernel
__global__ void matmulKernel(float* C, float* A, float* B, int rows, int cols, int k) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < rows && j < cols) {
        float sum = 0.0f;
        for (int p = 0; p < k; p++) {
            sum += A[i * k + p] * B[p * cols + j];
        }
        C[i * cols + j] = sum;
    }
}

// ReLU Kernel
__global__ void reluKernel(float* output, float* input, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        output[i] = input[i] > 0.0f ? input[i] : 0.0f;
    }
}

// MatMul Launch Function
void matmulGPU(float* h_C, float* h_A, float* h_B, int rows, int cols, int k) {
    float *d_A, *d_B, *d_C;
    
    cudaMalloc(&d_A, rows * k * sizeof(float));
    cudaMalloc(&d_B, k * cols * sizeof(float));
    cudaMalloc(&d_C, rows * cols * sizeof(float));
    
    cudaMemcpy(d_A, h_A, rows * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, k * cols * sizeof(float), cudaMemcpyHostToDevice);
    
    dim3 blockSize(16, 16);
    dim3 gridSize((cols + 15) / 16, (rows + 15) / 16);
    
    matmulKernel<<<gridSize, blockSize>>>(d_C, d_A, d_B, rows, cols, k);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    
    cudaMemcpy(h_C, d_C, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// ReLU Launch Function
void reluGPU(float* h_output, float* h_input, int size) {
    float *d_input, *d_output;
    
    cudaMalloc(&d_input, size * sizeof(float));
    cudaMalloc(&d_output, size * sizeof(float));
    
    cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (size + 255) / 256;
    
    reluKernel<<<gridSize, blockSize>>>(d_output, d_input, size);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    
    cudaMemcpy(h_output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(d_input);
    cudaFree(d_output);
}