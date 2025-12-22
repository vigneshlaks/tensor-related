#include "ops.h"

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


bool MatMulReluOp::verify(){
    return lhs->dimension[lhs->dimension.size() - 1] == rhs->dimension[rhs->dimension.size() - 2];
};

std::string MatMulReluOp::print() {
    return "MatMulRelu(" + std::to_string(lhs->dimension.size()) + ", " +
           std::to_string(rhs->dimension.size()) + " -> " +
           std::to_string(output->dimension.size()) + ")";
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

