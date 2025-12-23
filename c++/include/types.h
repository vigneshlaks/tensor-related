#ifndef TYPES_H
#define TYPES_H

#include <vector>
#include <string>

enum Precision {
    Int8,
    Float32
};

class Tensor {
private:
    // for now just store precision for kernel launches
    Precision precision = Float32;
    void fillStride();

public:
    std::vector<float> storage = {};  // Made public for direct access
    std::vector<size_t> dimension = {};
    std::vector<size_t> stride = {};

    Tensor(std::vector<size_t> d);
    Tensor(std::vector<size_t> d, std::vector<float> s);

    float getValue(std::vector<size_t> index);
    void setValue(std::vector<size_t> index, float value);
    void changePrecision(Precision p);
};

#endif