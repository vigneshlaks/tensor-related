#ifndef OPS_H
#define OPS_H

#include <vector>
#include <any>
#include <string>

#include "types.h"

class Op {
public:
    virtual ~Op() = default;
    virtual bool verify() = 0;
    virtual std::string print() = 0;
};

class MatMulOp : public Op {
public:
    Tensor* lhs;
    Tensor* rhs;
    Tensor* output;
    MatMulOp(Tensor* left, Tensor* right, Tensor* output) 
        : lhs(left), rhs(right), output(output) {}

    bool verify() override;
    std::string print() override;
};

class ReluOp : public Op {
public:
    Tensor* input;
    Tensor* output;
    ReluOp(Tensor* i, Tensor* o) : input(i), output(o) {}

    bool verify() override;
    std::string print() override;
};

class MatMulReluOp : public Op {
public:
    Tensor* lhs;
    Tensor* rhs;
    Tensor* output;
    MatMulReluOp(Tensor* lhs, Tensor* rhs, Tensor* output) 
        : lhs(lhs), rhs(rhs), output(output) {}

    bool verify() override;
    std::string print() override;
};

class MSEOp : public Op {
public:
    Tensor* input;
    Tensor* output;
    MSEOp(Tensor* i, Tensor* o) : input(i), output(o) {}

    bool verify() override;
    std::string print() override;
};

#endif