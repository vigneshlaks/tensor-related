#ifndef EXEC
#define EXEC

void matmul(float* h_C, float* h_A, float* h_B, int rows, int cols, int k);
void relu(float* h_output, float* h_input, int size);
void matmulRelu(float* h_C, float* h_A, float* h_B, int rows, int cols, int k);
void MSE(float* h_output, float* h_input, float* h_ground_truth, int size);
void quantization(float* h_output, float* h_input, int size);
void dequantization(float* h_output, float* h_input, int size);

#endif
