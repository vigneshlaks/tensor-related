#include "../include/types.h"
#include <iostream>

Tensor::Tensor(std::vector<size_t> d) : dimension(d) {
    fillStride();

    size_t totalSize = 1;
    for (size_t dim : dimension) {
        totalSize *= dim;
    }

    storage.resize(totalSize, 0.0f);
}

Tensor::Tensor(std::vector<size_t> d, std::vector<float> s) : dimension(d), storage(s) {
    fillStride();

    size_t totalSize = 1;
    for (size_t dim : dimension) {
        totalSize *= dim;
    }

    if (storage.size() != totalSize) {
        throw std::invalid_argument("Storage and Dimension Can't Be Reconciled");
    }
}

// gets the flat index representation
float Tensor::getValue(std::vector<size_t> index) {
    int flatIndex = 0;

    // bounds check
    for (int i = 0; i < dimension.size(); i++) {
        if (dimension[i] - 1 < index[i]) {
            throw std::invalid_argument("Invalid Tensor Indexing");
        }
    }

    for (int i = 0; i < dimension.size(); i++) {
        flatIndex += index[i] * stride[i];
    }

    return storage[flatIndex];
};

void Tensor::setValue(std::vector<size_t> index, float value) {
    int flatIndex = 0;

    // bounds check
    for (int i = 0; i < dimension.size(); i++) {
        dimension[i];
    }

    for (int i = 0; i < dimension.size(); i++) {
        flatIndex += index[i] * stride[i];
    }

    storage[flatIndex] = value;
};

void Tensor::fillStride() {
    stride.resize(dimension.size());
    stride.back() = 1;
    
    // specify jump sizes based on what comes after
    // consider what happens in 2D case
    // 
    for (int i = dimension.size() - 2; i >= 0; i--) {
        stride[i] = stride[i + 1] * dimension[i + 1];
    }
};

void Tensor::changePrecision(Precision p) {
    precision = p;
};

std::string Tensor::print() {
    std::string result = "[";
    for (size_t i = 0; i < dimension.size(); i++) {
        result += std::to_string(dimension[i]);
        if (i < dimension.size() - 1) result += "×";
    }
    result += ", " + std::string((precision == Float32) ? "Float32" : "Int8") + "]";
    return result;
};