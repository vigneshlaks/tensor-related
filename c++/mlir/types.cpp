#include "types.h"
#include <iostream>

bool SSA::name_set() {
    return !ssa.empty();
};

void SSA::set_name(std::string name) {
    ssa = name;
}


Tensor::Tensor(std::vector<size_t>& d) : dimension(d) {
    fillStride();
}

Tensor::Tensor(std::vector<size_t>& d, std::vector<float>& s) : dimension(d), storage(s) {
    fillStride();
}

float Tensor::getValue(std::vector<size_t> index) {
    int flatIndex = 0;
    
    for (int i = 0; i < dimension.size(); i++) {
        flatIndex += index[i] * stride[i];
    }

    return storage[flatIndex];
};

void Tensor::fillStride() {
    stride.resize(dimension.size());
    stride.back() = 1;
    
    // specify jump sizes based on what comes after
    // hence why we iterate backwards
    for (int i = dimension.size() - 2; i >= 0; i--) {
        stride[i] = stride[i + 1] * dimension[i + 1];
    }
};