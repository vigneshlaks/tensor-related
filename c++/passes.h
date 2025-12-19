
#ifndef PASSES_H
#define PASSES_H

#include <vector>
#include "frontend.h"
#include "ops.h"

class FusionPass
{
public:
    int apply(ComputeGraph* graph);

private:
    bool canFuse(Node* first, Node* second);

    void fuseNodes(ComputeGraph* graph, Node* first, Node* second);
};

class QuantizationPass
{
public:
    
};

struct Pass
{
    std::vector<Pattern*> patterns;
    Pass(std::vector<Pattern*> p) : patterns(p) {};
};

class PassManager
{
private:
    ComputeGraph computeGraph;
    std::vector<Pass*> passes;

public:
    PassManager(ComputeGraph cg, std::vector<Pass *> p) : computeGraph(cg), passes(p) {};
    void registerPass(Pass *pass);

    void run();

    bool verify();
};

#endif