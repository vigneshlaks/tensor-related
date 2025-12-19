#include "ops.h"
#include <sstream>

bool MatMulOp::verify() {
    if (lhs->dimension.size() == 1 || rhs->dimension.size() == 1){
        return false;
    }

    return lhs->dimension[lhs->dimension.size() - 1] == rhs->dimension[rhs->dimension.size() - 2];
};

std::string MatMulOp::print() {
    std::ostringstream oss;
    oss << "MatMul(" << lhs->dimension.size() << ", "
        << rhs->dimension.size() << " -> " << output->dimension.size() << ")";
    return oss.str();
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
    std::ostringstream oss;
    oss << "Relu(input, output -> " << input->dimension.size()
        << " -> " << output->dimension.size() << ")";
    return oss.str();
};


// same as matmul
bool MatMulReluOp::verify(){
    return lhs->dimension[lhs->dimension.size() - 1] == rhs->dimension[rhs->dimension.size() - 2];
};

std::string MatMulReluOp::print() {
    std::ostringstream oss;
    oss << "MatMulRelu(" << lhs->dimension.size() << ", "
        << rhs->dimension.size() << " -> " << output->dimension.size() << ")";
    return oss.str();
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
    std::ostringstream oss;
    oss << "MSE(input, output -> " << input->dimension.size()
        << " -> " << output->dimension.size() << ")";
    return oss.str();
};

