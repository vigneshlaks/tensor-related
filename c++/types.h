#ifndef TYPES_H
#define TYPES_H

#include <vector>
#include <string>

class Tensor {
private:
    std::vector<float> storage = {};
    void fillStride();

public:
    std::vector<size_t> dimension = {};
    std::vector<size_t> stride = {};

    Tensor(std::vector<size_t> d);
    Tensor(std::vector<size_t> d, std::vector<float> s);

    float getValue(std::vector<size_t> index);
};

#endif