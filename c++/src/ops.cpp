#include "../include/ops.h"

// compile flag to state if we have a GPU
// Not included in compilation if CUDA is not recognized
#ifdef CUDA_FOUND
#include "../include/gpu_exec.h"
#endif

#include <iostream>

int Op::setBackend(Backend b) {
    this->backend = b;
    return 0;
}

bool ConstOp::verify() {
    return true;
};

std::string ConstOp::print() {
    return "Const(output: " + output->print() + ")";
};

void ConstOp::execute() {
    return;
};


bool MatMulOp::verify() {
    if (lhs->dimension.size() == 1 || rhs->dimension.size() == 1) {
        return false;
    }

    return lhs->dimension[lhs->dimension.size() - 1] == rhs->dimension[rhs->dimension.size() - 2];
};

std::string MatMulOp::print() {
    return "MatMul(lhs: " + lhs->print() + ", rhs: " + rhs->print() + " → " + output->print() + ")";
};

void MatMulOp::execute() {
    if (backend == CPU) {
        // inline CPU for convenience
        // assume 2 dimensions for now

        for (size_t i = 0; i < output->dimension.at(0); i++) {
            for (size_t j = 0; j < output->dimension.at(1); j++) {
                for (size_t k = 0; k < rhs->dimension.at(1); k++) {

                    float lhs_val = lhs->getValue({i, k});
                    float rhs_val = rhs->getValue({k, j});
                    float curr_val = output->getValue({i, j});
                    float newValue = lhs_val * rhs_val + curr_val;
                    
                    output->setValue({i, j}, newValue);
                }
            }
        }
    } else {
        // compile flag to state if we have a GPU
        // GPU function is not compiled if CUDA is not recognized
        #ifdef CUDA_FOUND
        // Get the memory address for the first element
        float* h_C = &(output->storage[0]);
        float* h_A = &(lhs->storage[0]);
        float* h_B = &(rhs->storage[0]);
        matmul(h_C, h_A, h_B, output->dimension[0], output->dimension[1], rhs->dimension[1]);
        #else
            throw std::runtime_error("GPU Implementation Not Supported");
        #endif
    }
};

bool ReluOp::verify() {
    if (input->dimension.size() != output->dimension.size()) {
        return false;
    }

    // check same shape
    for (int i=0; i < input->dimension.size(); i++) {
        if (input->dimension[i] != output->dimension[i]) {
            return false;
        }
    }

    return true;
};

std::string ReluOp::print() {
    return "Relu(input: " + input->print() + " → " + output->print() + ")";
};

void ReluOp::execute() {
    if (backend == CPU) {
        for (size_t i = 0; i < output->dimension.at(0); i++) {
            for (size_t j = 0; j < output->dimension.at(1); j++) {
                if (input->getValue({i, j}) < 0) {
                    // update output value to zero
                    output->setValue({i, j}, 0);
                } else {
                    // set output value to be same as input
                    output->setValue({i,j}, input->getValue({i, j}));
                }
            }
        }
    } else {
        // compile flag to state if we have a GPU
        // GPU function is not compiled if CUDA is not recognized
        #ifdef CUDA_FOUND
        float* h_input = &(input->storage[0]);
        float* h_output = &(output->storage[0]);
        int size = input->storage.size();

        relu(h_output, h_input, size);
        #else
            throw std::runtime_error("GPU Implementation Not Supported");
        #endif
    }
};

bool MatMulReluOp::verify(){
    return lhs->dimension[lhs->dimension.size() - 1] == rhs->dimension[rhs->dimension.size() - 2];
};

std::string MatMulReluOp::print() {
    return "MatMulRelu(lhs: " + lhs->print() + ", rhs: " + rhs->print() + " → " + output->print() + ")";
};

void MatMulReluOp::execute() {
    if (backend == CPU) {
        // inline CPU for convenience
        // assume 2 dimensions for now
        for (size_t i = 0; i < output->dimension.at(0); i++) {
            for (size_t j = 0; j < output->dimension.at(1); j++) {
                // rhs->dimension.at(1) refers to the output column
                for (size_t k = 0; k < rhs->dimension.at(1); k++) {
                    // ith row and jth column accumulate
                    float newValue = lhs->getValue({i, k}) * rhs->getValue({k, j}) + output->getValue({i,j});
                    output->setValue({i, j}, newValue);
                }

                if (output->getValue({i, j}) < 0) {
                    output->setValue({i, j}, 0);
                }
            }
        }
    } else {
        #ifdef CUDA_FOUND
        float* h_C = &(output->storage[0]);
        float* h_A = &(lhs->storage[0]);
        float* h_B = &(rhs->storage[0]);
        matmulRelu(h_C, h_A, h_B, output->dimension[0], output->dimension[1], rhs->dimension[1]);
        #else
            throw std::runtime_error("GPU Implementation Not Supported");
        #endif
    }
};

bool MSEOp::verify(){
    if (input->dimension.size() != output->dimension.size() || input->dimension.size() != ground_truth->dimension.size()) {
        return false;
    }

    // check same shape
    for (int i=0; i < input->dimension.size(); i++) {
        if (input->dimension[i] != output->dimension[i] || input->dimension[i] != ground_truth->dimension[i]) {
            return false;
        }
    }

    return true;
};

std::string MSEOp::print() {
    return "MSE(input: " + input->print() + " → " + output->print() + ")";
};

void MSEOp::execute() {
    if (backend == CPU) {
        for (size_t i = 0; i < output->dimension.at(0); i++) {
            for (size_t j = 0; j < output->dimension.at(1); j++) {
                float diff = input->getValue({i, j}) - ground_truth->getValue({i, j});
                output->setValue({i, j}, diff * diff);
            }
        }
    } else {
        // compile flag to state if we have a GPU
        // GPU function is not compiled if CUDA is not recognized
        #ifdef CUDA_FOUND
        float* h_output = &(output->storage[0]);
        float* h_input = &(input->storage[0]);
        float* h_ground_truth = &(ground_truth->storage[0]);
        int size = input->storage.size();

        MSE(h_output, h_input, h_ground_truth, size);
        #else
            throw std::runtime_error("GPU Implementation Not Supported");
        #endif
    }
};

