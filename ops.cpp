#include "ops.h"
#include <sstream>

bool MatMulOp::verify(){
    return lhs.dimension[lhs.dimension.size() - 1] == rhs.dimension[rhs.dimension.size() - 2];
};

std::string MatMulOp::print() {
    std::ostringstream oss;
    oss << "lhs , rhs : " << lhs.dimension.size() << ", " << rhs.dimension.size() << " -> " << output.dimension.size();
    return oss.str();
};