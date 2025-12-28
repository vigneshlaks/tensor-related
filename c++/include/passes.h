
#ifndef PASSES_H
#define PASSES_H

#include <vector>
#include "frontend.h"
#include "ops.h"
#include "types.h"

class Pass {
public:
    virtual ~Pass() = default;
    virtual int globalApply(ComputeGraph* graph) = 0;
    virtual int localApply(ComputeGraph* graph) = 0;
};

// fuse operations
class FusionPass : public Pass {
private:
    bool canFuse(Node *first, Node *second);
    void fuseNodes(ComputeGraph *graph, Node *first, Node *second);
public:
    int globalApply(ComputeGraph* graph) override;
    int localApply(ComputeGraph* graph) override;
};

// change precision
class QuantizationPass : public Pass {
private:
    Precision precision;
public:
    QuantizationPass(Precision p) : precision(p) {};
    int globalApply(ComputeGraph* graph) override;
    int localApply(ComputeGraph* graph) override;
};

// choose a backend (cpu or gpu)
class BackendPass : public Pass {
private:
    Backend backend;
public:
    BackendPass(Backend b) : backend(b) {}
    int globalApply(ComputeGraph* graph) override;
    int localApply(ComputeGraph* graph) override;
};

class PassManager
{
private:
    ComputeGraph* computeGraph;
    std::vector<Pass*> passes;
public:
    PassManager(ComputeGraph* cg, std::vector<Pass*> p) : computeGraph(cg), passes(p) {};
    void registerPass(Pass *pass);

    void runLocal();
    void runGlobal();
    bool verify();
};

#endif