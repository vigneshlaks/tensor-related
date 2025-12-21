
#include "frontend.h"
#include <map>
#include <iostream>

// TODO
int parseBytecode()
{
    return 0;
};

ComputeGraph parseJSON(json instrs)
{
    Node* head = nullptr;

    // create start node
    Node* curr = new Node;
    curr->prev = head;

    // set to first node
    head = curr;

    // for deleting nodes in O(1)
    std::map<std::string, Node*> nodeMap;

    for (json instr : instrs)
    {
        std::string id = instr["id"].get<std::string>();
        OpType opType = instr["op"].get<OpType>();

        curr->id = id;
        curr->opType = opType;

        if (instr["op"] == "const")
        {
            auto storage = instr["value"].get<std::vector<float>>();
            auto dim = instr["dim"].get<std::vector<size_t>>();
            curr->output = std::make_shared<Tensor>(dim, storage);
            curr->operation = nullptr;
        }
        else if (instr["op"] == "matmul")
        {
            // get the input tensors
            std::shared_ptr<Tensor> lhs = nodeMap[instr["args"][0].get<std::string>()]->output;
            std::shared_ptr<Tensor> rhs = nodeMap[instr["args"][1].get<std::string>()]->output;
            // assumes 2 dimensions
            std::shared_ptr<Tensor> output = std::make_shared<Tensor>(std::vector<size_t>{lhs->dimension[0], rhs->dimension[1]});
            curr->output = output;
            curr->operation = std::make_unique<MatMulOp>(lhs, rhs, output);
        }
        else if (instr["op"] == "relu")
        {
            std::shared_ptr<Tensor> input = nodeMap[instr["args"][0].get<std::string>()]->output;
            // assumes 2 dimensions
            std::shared_ptr<Tensor> output = std::make_shared<Tensor>(std::vector<size_t>{input->dimension[0], input->dimension[1]});
            curr->output = output;
            curr->operation = std::make_unique<ReluOp>(input, output);
        }
        else if (instr["op"] == "mse_loss")
        {
            std::shared_ptr<Tensor> input = nodeMap[instr["args"][0].get<std::string>()]->output;
            // assumes 2 dimensions
            std::shared_ptr<Tensor> output = std::make_shared<Tensor>(std::vector<size_t>{input->dimension[0], input->dimension[1]});
            curr->output = output;
            curr->operation = std::make_unique<MSEOp>(input, output);
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
        nodeMap};
}
