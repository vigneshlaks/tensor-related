
#include "frontend.h"
#include <format>
#include <map>

// doubly linked list: https://docs.pytorch.org/docs/stable/fx.html
// since I'm only worred about inference right now, maybe it makes more sense to make it a singly linked list instead
ComputeGraph parseIR(json instrs) {
    Node* head = nullptr;
    
    // create start node
    Node* curr = new Node;
    curr->prev = head;
    
    // for deleting nodes in O(1)
    // maybe useful for other things as well
    std::map<std::string, Node*> nodeMap;

    for (json instr : instrs) {
        std::string id = instr["id"].get<std::string>();
        std::string opType = instr["op"].get<std::string>();

        curr->id = id;
        curr->opType = opType;

        if (instr["op"] == "const") {
            auto storage = instr["value"].get<std::vector<float>>();
            auto dim = instr["dim"].get<std::vector<size_t>>();
            Tensor* t = new Tensor(dim, storage);
            curr->output = t;
            curr->operation = nullptr;
        }
        else if (instr["op"] == "matmul") {
            // get the input tensors
            Tensor* lhs = nodeMap[instr["args"][0].get<std::string>()]->output;
            Tensor* rhs = nodeMap[instr["args"][1].get<std::string>()]->output;
            // assumes 2 dimensions
            Tensor* output = new Tensor({lhs->dimension[0], rhs->dimension[1]});
            curr->output = output;
            curr->operation = new MatMulOp(lhs, rhs, output);
        }
        else if (instr["op"] == "relu") {
            Tensor* input = nodeMap[instr["args"][0].get<std::string>()]->output;
            // assumes 2 dimensions
            Tensor* output = new Tensor({input->dimension[0], input->dimension[1]});
            curr->output = output;
            curr->operation = new ReluOp(input, output);
        }
        else if (instr["op"] == "mse_loss") {
            Tensor* input = nodeMap[instr["args"][0].get<std::string>()]->output;
            // assumes 2 dimensions
            Tensor* output = new Tensor({input->dimension[0], input->dimension[1]});
            curr->output = output;
            curr->operation = new ReluOp(input, output);
        }
        
        // assign prev pointer
        Node* next = new Node();
        next->prev = curr;
        curr->next = next;

        // move to next node
        curr = next;
    }
    
    return {
        head,
        nodeMap
    };
}

// TODO
void printComputeGraph(ComputeGraph cGraph) {
    return;
}