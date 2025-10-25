#ifndef TYPES_H
#define TYPES_H

#include <vector>
#include <string>

class SSA {
private:
    std::string ssa = "";

public:
    bool name_set();
    void set_name(std::string name);
};

class Tensor : SSA {
private:
    std::vector<float> storage = {};
    void fillStride();
    
public:
    std::vector<size_t> dimension = {};
    std::vector<size_t> stride = {};

    Tensor(std::vector<size_t>& d);
    Tensor(std::vector<size_t>& d, std::vector<float>& s);

    float getValue(std::vector<size_t> index);
};

#endif