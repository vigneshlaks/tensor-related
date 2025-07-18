// test.cpp
#include "passes.h"
#include <iostream>

int main() {
    // Create test tensors
    std::vector<size_t> dims1 = {2, 3};
    std::vector<size_t> dims2 = {3, 4};  
    std::vector<size_t> dims3 = {1, 4};

    Tensor t1(dims1), t2(dims2), t3(dims3), t4(dims3);
    
    // Create ops: matmul followed by relu (should fuse)
    std::vector<Op*> ops = {
        new MatMulOp(t1, t2, t3),
        new ReluOp(t3, t4)
    };
    
    // Create fusion pattern and pass
    MatMulReluFusion* fusion = new MatMulReluFusion();
    Pass* pass = new Pass({fusion});
    
    // Run pass manager
    PassManager pm(ops, {pass});
    pm.run();
    std::string boolean = pm.verify() ? "true" : "false";
    std::cout << "verify: " << boolean;

    return 0;
}