#ifndef TYPES_H
#define TYPES_H

#include <vector>

class Tensor {
private:
    std::vector<float> storage = {};
    void fillStride();
    
public:
    std::vector<size_t> dimension = {};
    std::vector<size_t> stride = {};

    Tensor(std::vector<size_t>& d);
    Tensor(std::vector<size_t>& d, std::vector<float>& s);

    // Get value at specific index
    float getValue(std::vector<size_t> index);
};

#endif