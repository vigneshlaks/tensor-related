// go from json to ir for now
#ifndef FRONTEND_H
#define FRONTEND_H

#include "ops.h"
#include "types.h"

#include <nlohmann/json.hpp>
using json = nlohmann::json;

enum OpType {
    Matmul,
    Relu,
    MatmulRelu,
    MSE
};

// IR Representation
struct Node {
    std::string id;
    OpType opType;
    std::shared_ptr<Tensor> output;
    std::unique_ptr<Op> operation;

    Node* prev;
    Node* next;
};

struct ComputeGraph {
    // just store the head for inference
    Node* head;
    std::map<std::string, Node*> nodeMap;
};

int parseBytecode();
ComputeGraph parseJSON(json inputIR);
void printComputeGraph(ComputeGraph cgraph);

#endif