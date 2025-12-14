#include "frontend.h"
#include <format>
#include <map>

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
    // just store the head
    Node* head;
    // would need tail if backprop ends up getting implemented
    std::map<std::string, Node*> nodeRegistry;
};

// doubly linked list: https://docs.pytorch.org/docs/stable/fx.html
std::vector<std::pair<Tensor, Op>> parseIR(json instrs) {
    Node* head = nullptr;
    Node* tail = nullptr;
    
    // go to any place in list
    std::map<std::string, Node*> nodeMap;

    for (json instr : instrs) {
        std::string id = instr["id"].get<std::string>();
        std::string opType = instr["op"].get<std::string>();

        Node* node = new Node();
        node->id = id;
        node->opType = opType;


        if (instr["op"] == "const") {
            auto storage = instr["value"].get<std::vector<float>>();
            auto dim = instr["dim"].get<std::vector<size_t>>();
            node->output = new Tensor(dim, storage);

        }
        // similar for rest
        // TODO: pointer logic
        else if (instr["op"] == "matmul") {

        }
        else if (instr["op"] == "relu") {

        }
        else if (instr["op"] == "mse_loss") {

        }
    }

    return res;
}