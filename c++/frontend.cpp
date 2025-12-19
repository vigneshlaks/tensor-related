
#include "frontend.h"
#include <map>
#include <iostream>

ComputeGraph parseJSON(json instrs) {
    Node* head = nullptr;
    
    // create start node
    Node* curr = new Node;
    curr->prev = head;
    
    // set to first node
    head = curr;
    
    // for deleting nodes in O(1)
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
            curr->operation = new MSEOp(input, output);
        }

        // store in map 
        nodeMap[instr["id"]] = curr;
        
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

void printComputeGraph(ComputeGraph cGraph) {
    std::cout << "=== Compute Graph ===" << std::endl;
    std::cout << "Total nodes: " << cGraph.nodeMap.size() << std::endl << std::endl;

    Node* curr = cGraph.head;
    int nodeIndex = 0;

    while (curr != nullptr && curr->id != "") {
        std::cout << "Node " << nodeIndex << ":" << std::endl;
        std::cout << "  ID: " << curr->id << std::endl;
        std::cout << "  Op Type: " << curr->opType << std::endl;

        if (curr->output != nullptr) {
            std::cout << "  Output Shape: [";
            for (size_t i = 0; i < curr->output->dimension.size(); i++) {
                std::cout << curr->output->dimension[i];
                if (i < curr->output->dimension.size() - 1) {
                    std::cout << ", ";
                }
            }
            std::cout << "]" << std::endl;
        }

        std::cout << std::endl;
        curr = curr->next;
        nodeIndex++;
    }

    std::cout << "=====================" << std::endl;
}