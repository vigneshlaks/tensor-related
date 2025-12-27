
#include "../include/frontend.h"
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
    std::unordered_map<std::string, Node*> nodeMap;

    std::unordered_map<std::string, OpType> opTypeMap = {
        {"const", OpType::Const},
        {"matmul", OpType::Matmul},
        {"relu", OpType::Relu},
        {"mse_loss", OpType::MSE}
    };

    for (json instr : instrs)
    {
        std::string id = instr["id"].get<std::string>();

        curr->id = id;
        curr->opType = opTypeMap[instr["op"].get<std::string>()];

        if (instr["op"] == "const")
        {
            auto storage = instr["value"].get<std::vector<float>>();
            auto dim = instr["dim"].get<std::vector<size_t>>();
            std::shared_ptr<Tensor> output = std::make_shared<Tensor>(dim, storage);

            curr->output = output;
            curr->operation = std::make_unique<ConstOp>(output);
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
            // get the input tensors
            std::shared_ptr<Tensor> input = nodeMap[instr["args"][0].get<std::string>()]->output;
            // assumes 2 dimensions
            std::shared_ptr<Tensor> output = std::make_shared<Tensor>(std::vector<size_t>{input->dimension[0], input->dimension[1]});
            curr->output = output;
            curr->operation = std::make_unique<ReluOp>(input, output);
        }
        else if (instr["op"] == "mse_loss")
        {
            // get the input tensors
            std::shared_ptr<Tensor> input = nodeMap[instr["args"][0].get<std::string>()]->output;
            std::shared_ptr<Tensor> output = std::make_shared<Tensor>(std::vector<size_t>{input->dimension[0], input->dimension[1]});

            // value and dim within the json for the loss represents the ground truth
            std::vector<size_t> dim = instr["dim"].get<std::vector<size_t>>();
            std::vector<float> storage = instr["value"].get<std::vector<float>>();
            std::shared_ptr<Tensor> ground_truth = std::make_shared<Tensor>(dim, storage);

            curr->operation = std::make_unique<MSEOp>(input, output, ground_truth);
        }

        // store in map
        nodeMap[instr["id"]] = curr;

        // assign pointers
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

// TODO make a better version of this and add graph viz
void printComputeGraph(ComputeGraph cgraph)
{
    std::map<OpType, std::string> opNames = {
        {OpType::Const, "Const"},
        {OpType::Matmul, "MatMul"},
        {OpType::Relu, "ReLU"},
        {OpType::MatmulRelu, "MatMul+ReLU"},
        {OpType::MSE, "MSE"}
    };

    std::cout << "Graph (" << cgraph.nodeMap.size() << " nodes):" << std::endl;

    Node* curr = cgraph.head;
    while (curr != nullptr && !curr->id.empty()) {
        std::cout << "  " << curr->id << " [" << opNames[curr->opType] << "]";

        if (curr->output) {
            std::cout << " → [";
            for (size_t i = 0; i < curr->output->dimension.size(); i++) {
                std::cout << curr->output->dimension[i];
                if (i < curr->output->dimension.size() - 1) std::cout << "×";
            }
            std::cout << "]";
        }

        if (curr->operation) {
            std::cout << " | Op: " << curr->operation->print();
        } else {
            std::cout << " | No operation (const)";
        }

        std::cout << " | Prev: " << (curr->prev ? curr->prev->id : "null");
        std::cout << " | Next: " << (curr->next && !curr->next->id.empty() ? curr->next->id : "null");

        std::cout << std::endl;
        curr = curr->next;
    }
}

// void printNode() {
// }
