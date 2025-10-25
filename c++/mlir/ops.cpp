#include "ops.h"
#include <sstream>
#include <format>

bool MatMulOp::verify() {
    if (lhs.dimension.size() == 1 || rhs.dimension.size() == 1){
        return false;
    }

    // TODO: check output
    return lhs.dimension[lhs.dimension.size() - 1] == rhs.dimension[rhs.dimension.size() - 2];
};

std::string MatMulOp::print() {
    return std::format("MatMul({}, {} -> {})", lhs.dimension.size(), 
                      rhs.dimension.size(), output.dimension.size());
};


bool ReluOp::verify() {
    if (input.dimension.size() != output.dimension.size()) {
        return false;
    }

    // check same shape
    for (int i=0; i < input.dimension.size(); i++) {
        if (input.dimension[i] != output.dimension[i]) {
            return false;
        }
    }

    return true;
};

std::string ReluOp::print() {
    return std::format("Relu(input, output -> {} -> {})", input.dimension.size(), output.dimension.size());
};


// same as matmul
bool MatMulReluOp::verify(){
    return lhs.dimension[lhs.dimension.size() - 1] == rhs.dimension[rhs.dimension.size() - 2];
};

std::string MatMulReluOp::print() {
    return std::format("MatMulRelu({}, {} -> {})", lhs.dimension.size(), 
                      rhs.dimension.size(), output.dimension.size());
};