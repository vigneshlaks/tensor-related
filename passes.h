
#ifndef PASSES_H
#define PASSES_H

#include <vector>
#include "ops.h"

struct Pattern {
    virtual bool matchRewrite(Op* op, std::vector<Op*>& context, size_t index) = 0; 
};

class MatMulReluFusion : public Pattern {
public:
    bool matchRewrite(Op* op, std::vector<Op*>& context, size_t index); 
};

struct Pass {
    std::vector<Pattern*> patterns;
    Pass(std::vector<Pattern*> p) : patterns(p) {};
};

class PassManager {
private:
    std::vector<Op*>& ops;
    std::vector<Pass*> passes;

public:
    PassManager(std::vector<Op*>& o, std::vector<Pass*> p) : ops(o), passes(p) {};
    void registerPass(Pass* pass);
    void addOperation(Op* op);
    
    // apply passes across ops
    void run();

    // run op verify
    bool verify();
};

#endif