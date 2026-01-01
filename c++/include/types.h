#ifndef TYPES_H
#define TYPES_H

#include <vector>
#include <string>

enum Precision {
    Int8,
    Float16,
    Float32
};

class Tensor {
private:
    void fillStride();

public:
    // for now just store precision for kernel launches
    Precision precision = Float32;
    
    std::vector<float> storage = {};
    std::vector<size_t> dimension = {};
    
    // stride depict how many elements in storage
    // are between contiguous elements along a specific dimension
    std::vector<size_t> stride = {};

    Tensor(std::vector<size_t> dimension);
    Tensor(std::vector<size_t> dimension, std::vector<float> storage);

    float getValue(std::vector<size_t> index);
    void setValue(std::vector<size_t> index, float value);
    void changePrecision(Precision p);
    std::string print();
    std::string printVerbose();
};

#endif