#ifndef OPS_GPU_H
#define OPS_GPU_H

void matmulGPU(float* h_C, float* h_A, float* h_B, int rows, int cols, int k);
void reluGPU(float* h_output, float* h_input, int size);

#endif