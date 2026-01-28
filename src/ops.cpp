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
    if (input == oldTensor) input = newTensor;                                                                           
    if (output == oldTensor) output = newTensor;
};

void DequantizationOp::updateTensorRefs(std::shared_ptr<Tensor> oldTensor, std::shared_ptr<Tensor> newTensor) {
    if (input == oldTensor) input = newTensor;                                                                           
    if (output == oldTensor) output = newTensor;
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
        // first dimension is batch for lhs 
        for (size_t i = 0; i < output->dimension.at(0); i++) {
            for (size_t j = 0; j < output->dimension.at(1); j++) {
                float sum = 0.0f;
                for (size_t k = 0; k < rhs->dimension.at(0); k++) {
                    sum += lhs->getValue({i, k}) * rhs->getValue({k, j});
                }
                output->setValue({i, j}, sum);
            }
        }
    } else {
        #ifdef CUDA_FOUND
            float* h_C = &(output->storage[0]);
            float* h_A = &(lhs->storage[0]);
            float* h_B = &(rhs->storage[0]);
            matmul(h_C, h_A, h_B, output->dimension[0], output->dimension[1], rhs->dimension[0]);
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
    if (lhs == oldTensor) lhs = newTensor;                                                                           
    if (rhs == oldTensor) rhs = newTensor;
    if (output == oldTensor) output = newTensor;
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
    // input and ground truth should be the same dimension
    if (input->dimension.size() != groundTruth->dimension.size()) {
        return false;
    }

    // check same shape
    for (int i = 0; i < input->dimension.size(); i++) {
        if (input->dimension[i] != groundTruth->dimension[i]) {
            return false;
        }
    }

    // output should be a scalar
    if (output->dimension.size() != 1) {
        return false;
    }

    return true;
};

std::string MSEOp::print() {
    return "MSE(input: " + input->print() + " → " + output->print() + ")";
};

void MSEOp::forward() {
    if (backend == CPU) {
        size_t batch = input->dimension[0];
        size_t features = input->dimension[1];

        float sum = 0.0f;
        for (size_t b = 0; b < batch; b++) {
            for (size_t f = 0; f < features; f++) {
                float diff = input->getValue({b, f}) - groundTruth->getValue({b, f});
                sum += diff * diff;
            }
        }

        float mse = sum / (batch * features);
        output->setValue({0}, mse);
    } else {
        #ifdef CUDA_FOUND
            float* h_output = &(output->storage[0]);
            float* h_input = &(input->storage[0]);
            float* h_groundTruth = &(h_groundTruth->storage[0]);
            int size = input->storage.size();

            MSE(h_output, h_input, h_groundTruth, size);
        #else
            throw std::runtime_error("GPU Implementation Not Supported");
        #endif
    }
};

void MSEOp::backward() {
    if (backend == CPU) {
        size_t batch = input->dimension[0];
        size_t features = input->dimension[1];
        float incomingGradient = output->getGrad({0});
        float scale = 2.0f / (batch * features);

        for (size_t b = 0; b < batch; b++) {
            for (size_t f = 0; f < features; f++) {
                float diff = input->getValue({b, f}) - groundTruth->getValue({b, f});
                input->accumulateGrad({b, f}, scale * diff * incomingGradient);
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

std::vector<size_t> ConstOp::inferOutputShape() {
    return output->dimension;
}

std::vector<size_t> MatMulOp::inferOutputShape() {
    if (!lhs || !rhs || lhs->dimension.size() < 2 || rhs->dimension.size() < 2) {
        return {};  // Invalid
    }

    std::vector<size_t> result_shape;

    // batching
    for (size_t i = 0; i < lhs->dimension.size() - 2; i++) {
        result_shape.push_back(lhs->dimension[i]);
    }

    size_t M = lhs->dimension[lhs->dimension.size() - 2];
    size_t N = rhs->dimension[rhs->dimension.size() - 1];
    result_shape.push_back(M);
    result_shape.push_back(N);

    return result_shape;
}

std::vector<size_t> ReluOp::inferOutputShape() {
    if (!input) {
        return {};
    }
    return input->dimension;
}

std::vector<size_t> MatMulReluOp::inferOutputShape() {
    if (!lhs || !rhs || lhs->dimension.size() < 2 || rhs->dimension.size() < 2) {
        return {};
    }

    std::vector<size_t> result_shape;

    // batching
    for (size_t i = 0; i < lhs->dimension.size() - 2; i++) {
        result_shape.push_back(lhs->dimension[i]);
    }

    size_t M = lhs->dimension[lhs->dimension.size() - 2];
    size_t N = rhs->dimension[rhs->dimension.size() - 1];
    result_shape.push_back(M);
    result_shape.push_back(N);

    return result_shape;
}

std::vector<size_t> QuantizationOp::inferOutputShape() {
    if (!input) {
        return {};
    }
    return input->dimension;
}

std::vector<size_t> DequantizationOp::inferOutputShape() {
    if (!input) {
        return {};
    }
    return input->dimension;
}

std::vector<size_t> MSEOp::inferOutputShape() {
    if (!input) {
        return {};
    }
    return input->dimension;
};

std::vector<size_t> SoftmaxOp::inferOutputShape() {
    if (!input) {
        return {};
    }
    return input->dimension;
};

std::vector<size_t> CrossEntropyOp::inferOutputShape() {
    if (!input) {
        return {};
    }
    return input->dimension;
};

bool SoftmaxOp::verify() {
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

std::string SoftmaxOp::print() {
    return "Softmax(input: " + input->print() + " → " + output->print() + ")";
};

void SoftmaxOp::forward() {
    if (backend == CPU) {
        // dim[0] = batch, dim[1] = classes
        size_t batch = input->dimension[0];
        size_t classes = input->dimension[1];

        for (size_t b = 0; b < batch; b++) {
            float max_val = -std::numeric_limits<float>::infinity();
            for (size_t c = 0; c < classes; c++) {
                max_val = std::max(max_val, input->getValue({b, c}));
            }

            // compute denominator for this row
            float denom = 0.0f;
            for (size_t c = 0; c < classes; c++) {
                denom += std::exp(input->getValue({b, c}) - max_val);
            }

            // compute softmax for this row
            for (size_t c = 0; c < classes; c++) {
                float val = std::exp(input->getValue({b, c}) - max_val) / denom;
                output->setValue({b, c}, val);
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

void SoftmaxOp::backward() {
    throw std::runtime_error("Softmax is an inference only operator");
};

void SoftmaxOp::updateTensorRefs(std::shared_ptr<Tensor> oldTensor, std::shared_ptr<Tensor> newTensor) {
    if (input == oldTensor) {
        input = newTensor;
    }
};

bool CrossEntropyOp::verify() {
    // input and ground truth should be the same shape
    if (input->dimension.size() != groundTruth->dimension.size()) {
        return false;
    }

    for (size_t i = 0; i < input->dimension.size(); i++) {
        if (input->dimension[i] != groundTruth->dimension[i]) {
            return false;
        }
    }

    // output should be a scalar
    if (output->dimension.size() != 1 || output->dimension[0] != 1) {
        return false;
    }

    return true;
};

std::string CrossEntropyOp::print() {
    return "CrossentropyOp(input: " + input->print() + " → " + output->print() + ")";
};

void CrossEntropyOp::forward() {
    size_t batch = input->dimension[0];
    size_t classes = input->dimension[1];

    float total_loss = 0.0f;
    for (size_t b = 0; b < batch; b++) {
        float sample_loss = 0.0f;
        for (size_t c = 0; c < classes; c++) {
            // the epsilon value 
            float epsilon = 1e-8f;
            float pred = input->getValue({b, c}) + epsilon;
            sample_loss += groundTruth->getValue({b, c}) * std::log(pred);
        }
        // sample across batches
        total_loss += -sample_loss;
    }

    output->setValue({0}, total_loss / batch);
};

void CrossEntropyOp::backward() {
    size_t batch = input->dimension[0];
    size_t classes = input->dimension[1];
    float incomingGrad = output->getGrad({0});

    for (size_t b = 0; b < batch; b++) {
        for (size_t c = 0; c < classes; c++) {
            float pred = input->getValue({b, c}) + 1e-8f;
            float grad = (-groundTruth->getValue({b, c}) / pred) * incomingGrad / batch;
            input->accumulateGrad({b, c}, grad);
        }
    }
};

void CrossEntropyOp::updateTensorRefs(std::shared_ptr<Tensor> oldTensor, std::shared_ptr<Tensor> newTensor) {
    if (input == oldTensor) input = newTensor;                                                                           
    if (output == oldTensor) output = newTensor;
};