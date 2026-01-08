#include "../include/passes.h"
#include <iostream>
#include <algorithm> 

int FusionPass::globalApply(LinkedList* list) {
    int fusionCount = 0;

    if (!list || !list->head) {
        return 0;
    }

    Node* current = list->head;

    while (current != nullptr && current->next != nullptr) {
        Node* next = current->next;

        if (canFuse(current, next)) {
            fuseNodes(list, current, next);
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
};

bool FusionPass::canFuse(Node* first, Node* second) {
    if (!first || !second) {
        return false;
    }

    if (first->opType == Matmul && second->opType == Relu) {
        return true;
    }

    return false;
};

// backend is associated with first node's backend
void FusionPass::fuseNodes(LinkedList* list, Node* first, Node* second) {
    // first and second node
    Node* newPrev = first->prev;
    Node* newNext = second->next;
    
    // construct node
    if (first->opType == Matmul && second->opType == Relu) {
        Node* fusedNode = new Node();
        Backend newBackend = first->operation->backend;
        
        std::shared_ptr<Tensor> lhs = dynamic_cast<MatMulOp*>(first->operation.get())->lhs;
        std::shared_ptr<Tensor> rhs = dynamic_cast<MatMulOp*>(first->operation.get())->rhs;
        std::shared_ptr<Tensor> output = dynamic_cast<ReluOp*>(second->operation.get())->output;
        std::string fusedId = first->id + "_" + second->id;
        
        fusedNode->output = output;
        fusedNode->operation = std::make_unique<MatMulReluOp>(lhs, rhs, output);
        fusedNode->operation->backend = newBackend;
        fusedNode->opType= MatmulRelu;
        fusedNode->id = fusedId;

        // rearrange pointers
        if (newPrev != nullptr) {
            newPrev->next = fusedNode;
        } else {
            // implies we're the new head
            list->head = fusedNode;
        }

        if (newNext != nullptr) {
            newNext->prev = fusedNode;
        }
        fusedNode->next = newNext;
        fusedNode->prev = newPrev;

        // get rid of old nodes
        list->nodeMap.erase(first->id);
        list->nodeMap.erase(second->id);
        delete first;
        delete second;

        // add to map
        list->nodeMap[fusedNode->id] = fusedNode;
    } else {
        throw std::invalid_argument("Unsupport Node Fusion");
    }
};

int PrecisionPass::globalApply(LinkedList* list) {
    Node* current = list->head;

    while (current != nullptr) {
        if (current->opType == Matmul) {
            MatMulOp* matmul = dynamic_cast<MatMulOp*>(current->operation.get());
            matmul->lhs->changePrecision(precision);
            matmul->rhs->changePrecision(precision);
            matmul->output->changePrecision(precision);
        } else if (current->opType == MatmulRelu) {
            MatMulReluOp* matmulRelu = dynamic_cast<MatMulReluOp*>(current->operation.get());
            matmulRelu->lhs->changePrecision(precision);
            matmulRelu->rhs->changePrecision(precision);
            matmulRelu->output->changePrecision(precision);
        }
        current = current->next;
    }
    return 0;
};

int QuantizationPass::globalApply(LinkedList* list) {
    Node* firstNode = list->head;
    if (firstNode->opType != Const) {
        return 0;
    }

    Node* quantNode = new Node();
    quantNode->id = firstNode->id + "_quantized";
    quantNode->opType = Const;

    std::shared_ptr<Tensor> input = firstNode->output;

    quantNode->output = input;
    quantNode->operation = std::make_unique<QuantizationOp>(input, input);
    QuantizationOp* quantOp = dynamic_cast<QuantizationOp*>(quantNode->operation.get());
    quantOp->precision = precision;

    // Use same backend as next node
    if (firstNode->next && firstNode->next->operation) {
        quantOp->backend = firstNode->next->operation->backend;
    }

    // Place in node between first and next
    Node* nextNode = firstNode->next;
    firstNode->next = quantNode;
    quantNode->prev = firstNode;
    quantNode->next = nextNode;
    if (nextNode) {
        nextNode->prev = quantNode;
    }

    // Add to nodemap
    list->nodeMap[quantNode->id] = quantNode;

    // We need to dequantize before computing the loss
    Node* lastComputeNode = list->head;
    while (lastComputeNode->next != nullptr && !lastComputeNode->next->id.empty()) {
        if (lastComputeNode->next->opType == MSE) {
            break;
        }
        lastComputeNode = lastComputeNode->next;
    }

    // Insert dequantization node before loss if loss exists
    if (lastComputeNode && lastComputeNode->next && lastComputeNode->next->opType == MSE) {
        Node* dequantNode = new Node();
        dequantNode->id = lastComputeNode->id + "_dequantized";
        dequantNode->opType = OpType::Const;

        // Dequantization is done in-place (same as quantization for consistency)
        std::shared_ptr<Tensor> output = lastComputeNode->output;

        // Pass quantOp pointer so dequantization can access the scale factor
        dequantNode->output = output;
        dequantNode->operation = std::make_unique<DequantizationOp>(output, output, quantOp);
        DequantizationOp* dequantOp = dynamic_cast<DequantizationOp*>(dequantNode->operation.get());

        // Use same backend as previous node
        if (lastComputeNode->operation) {
            dequantOp->backend = lastComputeNode->operation->backend;
        }

        // Place in node
        Node* mseNode = lastComputeNode->next;
        lastComputeNode->next = dequantNode;
        dequantNode->prev = lastComputeNode;
        dequantNode->next = mseNode;
        mseNode->prev = dequantNode;
        
        // Register the dequantization node in the map
        list->nodeMap[dequantNode->id] = dequantNode;
    }

    return 1;
}

int BackendPass::globalApply(LinkedList* list) {
    Node* current = list->head;

    while (current != nullptr) {
        // const has null operation
        if (current->operation != nullptr) {
            
            // set the backend associated with this pass to the operation
            current->operation->setBackend(this->backend);
        }
        current = current->next;
    }

    return 0;
}

void PassManager::registerPass(Pass* pass) {
    passes.push_back(pass);
};

// run global runs full optimization one by one
void PassManager::runGlobal() {
    for (Pass* pass : passes) {
       pass->globalApply(linkedList);
    }
};

bool PassManager::verify() {
    if (!linkedList || !linkedList->head) {
        return false;
    }

    Node* current = linkedList->head;
    Node* prev = nullptr;

    while (current != nullptr && !current->id.empty()) {
        // Check prev pointer consistency
        if (current->prev != prev) {
            return false;
        }

        // Verify operation if it exists
        if (current->operation != nullptr) {
            if (!current->operation->verify()) {
                return false;
            }
        }

        prev = current;
        current = current->next;
    }

    return true;
};
