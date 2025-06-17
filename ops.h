#ifndef OPS_H
#define OPS_H

#include <vector>
#include <any>
#include <string>
#include <format>

#include "types.h"

class Op {
public:
    virtual ~Op() = default;
    virtual bool verify() = 0;
    virtual std::string print() = 0;
};

class MatMulOp : public Op {
private:
    Tensor lhs, rhs;
    Tensor output;

public:
    MatMulOp(const Tensor& left, const Tensor& right, const Tensor& output) 
        : lhs(left), rhs(right), output(output) {}

    bool verify() override;
    std::string print() override;
};

#endif