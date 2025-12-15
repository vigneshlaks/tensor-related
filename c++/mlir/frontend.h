// go from json to ir for now
#ifndef FRONTEND_H
#define FRONTEND_H

#include "ops.h"
#include "types.h"

#include <nlohmann/json.hpp>
using json = nlohmann::json;

// IR Representation
struct Node {
    std::string id;
    std::string opType;
    std::vector<std::string> inputIds;
    Tensor* output;
    Op* operation;

    Node* prev;
    Node* next;
};

struct ComputeGraph {
    // just store the head for inference
    Node* head;
    std::map<std::string, Node*> nodeMap;
};

ComputeGraph parseIR(json inputIR);
void printComputeGraph(ComputeGraph cgraph);

#endif