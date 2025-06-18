#include "passes.h"
#include <iostream>

bool MatMulReluFusion::matchRewrite(Op* op, std::vector<Op*>& context, size_t index) {
  if (auto matmul = dynamic_cast<MatMulOp*>(op)) {
    // inbounds and next op is relu
    if (index + 1 < context.size() && dynamic_cast<ReluOp*>(context[index+1]) != nullptr) {
      MatMulReluOp* matmulrelu = new MatMulReluOp(matmul->lhs,matmul->rhs,  matmul->output);
      // swap
      context[index] = matmulrelu;
      //std::cout << static_cast <MatMulReluOp*> (context[index]) ->print();
      // delete
      context[index+1] = nullptr;
      //std::cout << context[index+1];
      return true;
    }
  }

  return false;
};

void PassManager::registerPass(Pass* pass) {
    passes.push_back(pass);
};

void PassManager::addOperation(Op* op) {
  ops.push_back(op);
};

void PassManager::run() {
  // passes applied sequentially
  for (auto& pass : passes) {
    std::vector<Pattern*> patterns = pass->patterns;

    for (int i = 0; i < ops.size(); i++) {
      for (int j = 0; j < patterns.size(); j++) {
        patterns[j]->matchRewrite(ops[i], ops, i);
      }
    }

    // moves nulls to end then remove them
    ops.erase(std::remove(ops.begin(), ops.end(), nullptr), ops.end());
  }
};
