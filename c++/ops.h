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
    std::shared_ptr<Tensor> lhs;
    std::shared_ptr<Tensor> rhs;
    std::shared_ptr<Tensor> output;
    MatMulOp(std::shared_ptr<Tensor> left, std::shared_ptr<Tensor> right, std::shared_ptr<Tensor> output) 
        : lhs(left), rhs(right), output(output) {}

    bool verify() override;
    std::string print() override;
};

class ReluOp : public Op {
public:
    std::shared_ptr<Tensor> input;
    std::shared_ptr<Tensor> output;
    ReluOp(std::shared_ptr<Tensor> i, std::shared_ptr<Tensor> o) : input(i), output(o) {}

    bool verify() override;
    std::string print() override;
};

class MatMulReluOp : public Op {
public:
    std::shared_ptr<Tensor> lhs;
    std::shared_ptr<Tensor> rhs;
    std::shared_ptr<Tensor> output;
    MatMulReluOp(std::shared_ptr<Tensor> lhs, std::shared_ptr<Tensor> rhs, std::shared_ptr<Tensor> output) 
        : lhs(lhs), rhs(rhs), output(output) {}

    bool verify() override;
    std::string print() override;
};

class MSEOp : public Op {
public:
    std::shared_ptr<Tensor> input;
    std::shared_ptr<Tensor> output;
    MSEOp(std::shared_ptr<Tensor> i, std::shared_ptr<Tensor> o) : input(i), output(o) {}

    bool verify() override;
    std::string print() override;
};

#endif