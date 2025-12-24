#include "../include/ops.h"
#include "../include/ops_gpu.h"

// Stub implementation when CUDA is not available
void matmulGPU(float* h_C, float* h_A, float* h_B, int rows, int cols, int k) {
    throw std::runtime_error("GPU support not available - CUDA not found");
}

int Op::setBackend(Backend b) {
    this->backend = b;
    return 0;
}

bool MatMulOp::verify() {
    if (lhs->dimension.size() == 1 || rhs->dimension.size() == 1) {
        return false;
    }

    return lhs->dimension[lhs->dimension.size() - 1] == rhs->dimension[rhs->dimension.size() - 2];
};

std::string MatMulOp::print() {
    return "MatMul(" + std::to_string(lhs->dimension.size()) + ", " +
           std::to_string(rhs->dimension.size()) + " -> " +
           std::to_string(output->dimension.size()) + ")";
};

void MatMulOp::execute() {
    if (backend == CPU) {
        // inline CPU for convenience
        // assume 2 dimensions for now
        for (size_t i = 0; i < output->dimension.at(0); i++) {
            for (size_t j = 0; j < output->dimension.at(1); j++) {
                // rhs->dimension.at(1) refers to the output column
                for (size_t k = 0; k < rhs->dimension.at(1); k++) {
                    // ith row and jth column accumulate
                    float newValue = lhs->getValue({i, k}) * rhs->getValue({k, j}) + output->getValue({i,j});
                    output->setValue({i,j}, newValue);
                }
            }
        }
    } else {
        // launch cuda kernel
        // gpu case assume not implemented
        float* h_C = &(output->storage[0]);
        float* h_A = &(lhs->storage[0]);
        float* h_B = &(rhs->storage[0]);
        matmulGPU(h_C, h_A, h_B, output->dimension[0], output->dimension[1], rhs->dimension[1]);
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
    return "Relu(input, output -> " + std::to_string(input->dimension.size()) +
           " -> " + std::to_string(output->dimension.size()) + ")";
};

void ReluOp::execute() {
    if (backend == CPU) {
        for (size_t i = 0; i < output->dimension.at(0); i++) {
            for (size_t j = 0; j < output->dimension.at(1); j++) {
                if (input->getValue({i, j}) < 0) {
                    output->setValue({i, j}, 0);
                }
            }
        }
    } else {
        throw std::runtime_error("GPU implementation not implemented");
    }
};

bool MatMulReluOp::verify(){
    return lhs->dimension[lhs->dimension.size() - 1] == rhs->dimension[rhs->dimension.size() - 2];
};

std::string MatMulReluOp::print() {
    return "MatMulRelu(" + std::to_string(lhs->dimension.size()) + ", " +
           std::to_string(rhs->dimension.size()) + " -> " +
           std::to_string(output->dimension.size()) + ")";
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
        throw std::runtime_error("GPU implementation not implemented");
    }
};

bool MSEOp::verify(){
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

std::string MSEOp::print() {
    return "MSE(input, output -> " + std::to_string(input->dimension.size()) +
           " -> " + std::to_string(output->dimension.size()) + ")";
};

void MSEOp::execute() {
    return;
};
