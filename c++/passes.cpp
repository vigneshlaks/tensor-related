#include "passes.h"
#include <iostream>
#include <algorithm>

int FusionPass::apply(ComputeGraph* graph) {
    int fusionCount = 0;

    if (!graph || !graph->head) {
        return 0;
    }

    Node* current = graph->head;

    while (current != nullptr && current->next != nullptr) {
        Node* next = current->next;

        if (canFuse(current, next)) {
            fuseNodes(graph, current, next);
            fusionCount++;

            if (current->next != nullptr) {
                current = current->next;
            } else {
                break;
            }
        } else {
            current = next;
        }
    }

    return fusionCount;
}

bool FusionPass::canFuse(Node* first, Node* second) {
    if (!first || !second) {
        return false;
    }

    if (first->opType == "matmul" && second->opType == "relu") {
        if (second->inputIds.size() == 1 &&
            second->inputIds[0] == first->id) {
            return true;
        }
    }

    

    return false;
}

void FusionPass::fuseNodes(ComputeGraph* graph, Node* first, Node* second) {
    // Create fused operation: MatMul + ReLU -> MatMulReLU
    if (first->opType == "matmul" && second->opType == "relu") {
        MatMulOp* matmulOp = dynamic_cast<MatMulOp*>(first->operation);

        if (matmulOp) {
            // Create fused operation
            // Use second's output tensor (ReLU's output) as the final output
            MatMulReluOp* fusedOp = new MatMulReluOp(
                matmulOp->lhs,
                matmulOp->rhs,
                second->output  // Use ReLU's output
            );

            // Update first node to be the fused node
            first->operation = fusedOp;
            first->opType = "matmul_relu";
            first->output = second->output;
            first->id = second->id;  // Take the output node's ID

            // Remove second node from the linked list
            first->next = second->next;
            if (second->next != nullptr) {
                second->next->prev = first;
            }

            // Update node map to remove old nodes and add fused node
            graph->nodeMap.erase(second->id);
            // The fused node keeps the second (output) node's ID
            graph->nodeMap[first->id] = first;

            // Clean up the old ReLU node
            delete second->operation;
            delete second;

            std::cout << "Fused MatMul + ReLU -> MatMulReLU" << std::endl;
        }
    }
}

void PassManager::registerPass(Pass* pass) {
    passes.push_back(pass);
};

void PassManager::run() {

};

bool PassManager::verify() {

};
