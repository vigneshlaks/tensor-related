
#ifndef PASSES_H
#define PASSES_H

#include <vector>
#include "frontend.h"
#include "ops.h"
#include <format>
#include "types.h"

class Pass {
public:
    virtual ~Pass() = default;
    virtual int apply(ComputeGraph *graph) = 0;
};

class FusionPass
{
public:
    int apply(ComputeGraph *graph);

private:
    bool canFuse(Node *first, Node *second);

    void fuseNodes(ComputeGraph *graph, Node *first, Node *second);
};

class QuantizationPass
{
    int apply(ComputeGraph *graph, std::variant<int, float>);
};

class PassManager
{
private:
    ComputeGraph computeGraph;
    std::vector<Pass *> passes;

public:
    PassManager(ComputeGraph cg, std::vector<Pass *> p) : computeGraph(cg), passes(p) {};
    void registerPass(Pass *pass);

    void run();

    bool verify();
};

#endif