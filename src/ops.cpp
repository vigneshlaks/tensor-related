#include "../include/ops.h"

// compile flag to state if we have a GPU
// not included in compilation if CUDA is not recognized
// used in other places as well
#ifdef CUDA_FOUND
#include "../include/gpu_exec.h"
#endif

#include <iostream>
#include <limits>
#include <algorithm>
#include <stdexcept>
#include <cmath>

namespace {
    std::shared_ptr<Tensor> matrixTranspose(std::shared_ptr<Tensor> input) {
        int rows = input->dimension.at(input->dimension.size() - 1);
        int cols = input->dimension.at(input->dimension.size() - 2);
        
        std::shared_ptr<Tensor> output = std::make_shared<Tensor>(std::vector<size_t>(rows, cols));
        
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                output->setValue({j, i}, input->getValue({i, j}));
            }
        }
        
        return output;
    }
};

int Op::setBackend(Backend b) {
    this->backend = b;
    return 0;
};

bool ConstOp::verify() {
    return true;
};

std::string ConstOp::print() {
    return "Const(output: " + output->print() + ")";
};

void ConstOp::forward() {
    return;
};

void ConstOp::backward() {
    return;
};

void ConstOp::updateTensorRefs(std::shared_ptr<Tensor> oldTensor, std::shared_ptr<Tensor> newTensor) {
    return;
};

bool QuantizationOp::verify() {
    // check if they are not the same size or equal to one
    if (input->dimension.size() != output->dimension.size() || input->dimension.size() == 1) {
        return false;
    }

    // check dimensions are the same
    for (int i = 0; i < input->dimension.size(); i++) {
        if (input->dimension[i] != output->dimension[i]) {
            return false;
        }
    }

    return true;
};

std::string QuantizationOp::print() {
    return "QuantizationOp(input: " + input->print() + " → " + output->print() + ")";
};

void QuantizationOp::forward() {
    if (backend == CPU) {
        // find max and min in tensor
        float x_max = -std::numeric_limits<float>::infinity();
        float x_min = std::numeric_limits<float>::infinity();

        for (float element : input->storage) {
            if (element > x_max) {
                x_max = element;
            }
            if (element < x_min) {
                x_min = element;
            }
        }

        float abs_x_min = std::abs(x_min);
        float abs_x_max = std::abs(x_max);

        int alpha, beta;
        if (precision == Float16) {
            alpha = 65504;
            beta = -65504;
        } else if (precision == Int8) {
            alpha = 127;
            beta = -127;
        } else {
            throw std::invalid_argument("Specified Precision not supported");
        }

        scale = (std::max(abs_x_min, abs_x_max) * 2) / (alpha - beta);
        for (int i = 0; i < input->storage.size(); i++) {
            float new_element = std::round(input->storage.at(i) / scale);

            // clipping
            if (new_element > alpha) {
                new_element = alpha;
            }
            if (new_element < beta) {
                new_element = beta;
            }

            output->storage.at(i) = new_element;
        }
    } else {
        #ifdef CUDA_FOUND
            // Get the memory address for the first element
            float* h_C = &(output->storage[0]);
            float* h_A = &(lhs->storage[0]);
            float* h_B = &(rhs->storage[0]);
            quantization(h_C, h_A, h_B, output->dimension[0], output->dimension[1], rhs->dimension[1]);
        #else
            throw std::runtime_error("GPU Implementation Not Supported");
        #endif
    }

    return;
};

void QuantizationOp::backward() {
    if (backend == CPU) {
        // Copy gradients through unchanged
        for (int i = 0; i < output->grad.size(); i++) {
            input->grad[i] = output->grad[i];
        }
    } else {
        #ifdef CUDA_FOUND
            throw std::runtime_error("GPU Implementation Not Supported");
        #else
            throw std::runtime_error("GPU Implementation Not Supported");
        #endif
    }
};

bool DequantizationOp::verify() {
    // check if they are not the same size or equal to one
    if (input->dimension.size() != output->dimension.size() || input->dimension.size() == 1) {
        return false;
    }

    // check dimensions are the same
    for (int i = 0; i < input->dimension.size(); i++) {
        if (input->dimension[i] != output->dimension[i]) {
            return false;
        }
    }

    return true;
};

std::string DequantizationOp::print() {
    return "QuantizationOp(input: " + input->print() + " → " + output->print() + ")";
};

void DequantizationOp::forward() {
    if (backend == CPU) {
        float scale = (quantOp != nullptr) ? quantOp->scale : 1.0f;
        for (int i = 0; i < input->storage.size(); i++) {
            output->storage.at(i) = input->storage.at(i) * scale;
        }
    } else {
        throw std::runtime_error("GPU Implementation Not Supported");
    }
};

void DequantizationOp::backward() {
    if (backend == CPU) {
        // Straight-through: pass gradients through unchanged
        for (int i = 0; i < output->grad.size(); i++) {
            input->grad[i] = output->grad[i];
        }
    } else {
        #ifdef CUDA_FOUND
            throw std::runtime_error("GPU Implementation Not Supported");
        #else
            throw std::runtime_error("GPU Implementation Not Supported");
        #endif
    }
};

void QuantizationOp::updateTensorRefs(std::shared_ptr<Tensor> oldTensor, std::shared_ptr<Tensor> newTensor) {
    if (input == oldTensor) {
        input = newTensor;
    }
};

void DequantizationOp::updateTensorRefs(std::shared_ptr<Tensor> oldTensor, std::shared_ptr<Tensor> newTensor) {
    if (input == oldTensor) {
        input = newTensor;
    }
};

bool MatMulOp::verify() {
    if (lhs->dimension.size() == 1 || rhs->dimension.size() == 1) {
        return false;
    }

    if (lhs->dimension[lhs->dimension.size() - 1] != rhs->dimension[rhs->dimension.size() - 2]) {
        return false;
    }

    return true;
};

std::string MatMulOp::print() {
    return "MatMul(lhs: " + lhs->print() + ", rhs: " + rhs->print() + " → " + output->print() + ")";
};

void MatMulOp::forward() {
    if (backend == CPU) {
        // inline CPU for convenience
        // assume 2 dimensions for now
        for (size_t i = 0; i < output->dimension.at(0); i++) {
            for (size_t j = 0; j < output->dimension.at(1); j++) {
                // rhs->dimension.at(0) refers to the intermediate column
                for (size_t k = 0; k < rhs->dimension.at(0); k++) {
                    float lhs_val = lhs->getValue({i, k});
                    float rhs_val = rhs->getValue({k, j});
                    float curr_val = output->getValue({i, j});
                    float newValue = lhs_val * rhs_val + curr_val;

                    output->setValue({i, j}, newValue);
                }
            }
        }
    } else {
        #ifdef CUDA_FOUND
            float* h_C = &(output->storage[0]);
            float* h_A = &(lhs->storage[0]);
            float* h_B = &(rhs->storage[0]);
            matmul(h_C, h_A, h_B, output->dimension[0], output->dimension[1], rhs->dimension[1]);
        #else
            throw std::runtime_error("GPU Implementation Not Supported");
        #endif
    }
};

void MatMulOp::backward() {
    if (backend == CPU) {
        // grad_lhs = output->grad @ rhs^T
        for (size_t i = 0; i < lhs->dimension[0]; i++) {
            for (size_t k = 0; k < lhs->dimension[1]; k++) {
                float sum = 0.0f;
                for (size_t j = 0; j < rhs->dimension[1]; j++) {
                    sum += output->getGrad({i, j}) * rhs->getValue({k, j});
                }
                lhs->accumulateGrad({i, k}, sum);
            }
        }

        // grad_rhs = lhs^T @ output->grad
        for (size_t k = 0; k < rhs->dimension[0]; k++) {
            for (size_t j = 0; j < rhs->dimension[1]; j++) {
                float sum = 0.0f;
                for (size_t i = 0; i < lhs->dimension[0]; i++) {
                    sum += lhs->getValue({i, k}) * output->getGrad({i, j});
                }
                rhs->accumulateGrad({k, j}, sum);
            }
        }
    } else {
        #ifdef CUDA_FOUND
            throw std::runtime_error("GPU Implementation Not Supported");
        #else
            throw std::runtime_error("GPU Implementation Not Supported");
        #endif
    }
};

void MatMulOp::updateTensorRefs(std::shared_ptr<Tensor> oldTensor, std::shared_ptr<Tensor> newTensor) {
    if (lhs == oldTensor) {
        lhs = newTensor;
    }
    if (rhs == oldTensor) {
        rhs = newTensor;
    }
};

bool ReluOp::verify() {
    // check if they are not the same size or equal to one
    if (input->dimension.size() != output->dimension.size() || input->dimension.size() == 1) {
        return false;
    }

    // check same shape
    for (int i = 0; i < input->dimension.size(); i++) {
        if (input->dimension[i] != output->dimension[i]) {
            return false;
        }
    }

    return true;
};

std::string ReluOp::print() {
    return "Relu(input: " + input->print() + " → " + output->print() + ")";
};

void ReluOp::forward() {
    if (backend == CPU) {
        for (size_t i = 0; i < output->dimension.at(0); i++) {
            for (size_t j = 0; j < output->dimension.at(1); j++) {
                if (input->getValue({i, j}) < 0) {
                    // update output value to zero
                    output->setValue({i, j}, 0);
                } else {
                    // set output value to be same as input
                    output->setValue({i, j}, input->getValue({i, j}));
                }
            }
        }
    } else {
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

void ReluOp::backward() {
    if (backend == CPU) {
        for (int i = 0; i < output->grad.size(); i++) {
            float mask = (output->storage[i] > 0) ? 1.0f : 0.0f;
            input->grad[i] += output->grad[i] * mask;
        }
    } else {
        #ifdef CUDA_FOUND
            throw std::runtime_error("GPU Implementation Not Supported");
        #else
            throw std::runtime_error("GPU Implementation Not Supported");
        #endif
    }
};

void ReluOp::updateTensorRefs(std::shared_ptr<Tensor> oldTensor, std::shared_ptr<Tensor> newTensor) {
    if (input == oldTensor) {
        input = newTensor;
    }
};

bool MatMulReluOp::verify() {
    return lhs->dimension[lhs->dimension.size() - 1] == rhs->dimension[rhs->dimension.size() - 2];
};

std::string MatMulReluOp::print() {
    return "MatMulRelu(lhs: " + lhs->print() + ", rhs: " + rhs->print() + " → " + output->print() + ")";
};

void MatMulReluOp::forward() {
    if (backend == CPU) {
        // inline CPU for convenience
        // assume 2 dimensions for now
        for (size_t i = 0; i < output->dimension.at(0); i++) {
            for (size_t j = 0; j < output->dimension.at(1); j++) {
                // rhs->dimension.at(0) refers to the intermediate column
                for (size_t k = 0; k < rhs->dimension.at(0); k++) {
                    // ith row and jth column accumulate
                    float newValue = lhs->getValue({i, k}) * rhs->getValue({k, j}) + output->getValue({i, j});
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

void MatMulReluOp::backward() {
    if (backend == CPU) {
        // grad_lhs = (output->grad * relu_mask) @ rhs^T
        for (size_t i = 0; i < lhs->dimension[0]; i++) {
            for (size_t k = 0; k < lhs->dimension[1]; k++) {
                float sum = 0.0f;
                for (size_t j = 0; j < rhs->dimension[1]; j++) {
                    float reluGrad = (output->getValue({i, j}) > 0) ? 1.0f : 0.0f;
                    sum += output->getGrad({i, j}) * reluGrad * rhs->getValue({k, j});
                }
                lhs->accumulateGrad({i, k}, sum);
            }
        }

        // grad_rhs = lhs^T @ (output->grad * relu_mask)
        for (size_t k = 0; k < rhs->dimension[0]; k++) {
            for (size_t j = 0; j < rhs->dimension[1]; j++) {
                float sum = 0.0f;
                for (size_t i = 0; i < lhs->dimension[0]; i++) {
                    float reluGrad = (output->getValue({i, j}) > 0) ? 1.0f : 0.0f;
                    sum += lhs->getValue({i, k}) * output->getGrad({i, j}) * reluGrad;
                }
                rhs->accumulateGrad({k, j}, sum);
            }
        }
    } else {
        #ifdef CUDA_FOUND
            throw std::runtime_error("GPU Implementation Not Supported");
        #else
            throw std::runtime_error("GPU Implementation Not Supported");
        #endif
    }
};

void MatMulReluOp::updateTensorRefs(std::shared_ptr<Tensor> oldTensor, std::shared_ptr<Tensor> newTensor) {
    if (lhs == oldTensor) {
        lhs = newTensor;
    }
    if (rhs == oldTensor) {
        rhs = newTensor;
    }
};

bool MSEOp::verify() {
    if (input->dimension.size() != output->dimension.size() || input->dimension.size() != ground_truth->dimension.size()) {
        return false;
    }

    // check same shape
    for (int i = 0; i < input->dimension.size(); i++) {
        if (input->dimension[i] != output->dimension[i] || input->dimension[i] != ground_truth->dimension[i]) {
            return false;
        }
    }

    return true;
};

std::string MSEOp::print() {
    return "MSE(input: " + input->print() + " → " + output->print() + ")";
};

void MSEOp::forward() {
    if (backend == CPU) {
        for (size_t i = 0; i < output->dimension.at(0); i++) {
            for (size_t j = 0; j < output->dimension.at(1); j++) {
                float diff = input->getValue({i, j}) - ground_truth->getValue({i, j});
                output->setValue({i, j}, diff * diff);
            }
        }
    } else {
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

void MSEOp::backward() {
    if (backend == CPU) {
        for (size_t i = 0; i < input->dimension[0]; i++) {
            for (size_t j = 0; j < input->dimension[1]; j++) {
                float grad = 2.0f * (input->getValue({i, j}) - ground_truth->getValue({i, j}));
                input->accumulateGrad({i, j}, grad * output->getGrad({i, j}));
            }
        }
    } else {
        #ifdef CUDA_FOUND
            throw std::runtime_error("GPU Implementation Not Supported");
        #else
            throw std::runtime_error("GPU Implementation Not Supported");
        #endif
    }
};

void MSEOp::updateTensorRefs(std::shared_ptr<Tensor> oldTensor, std::shared_ptr<Tensor> newTensor) {
    if (input == oldTensor) {
        input = newTensor;
    }
};
