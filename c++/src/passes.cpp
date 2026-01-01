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

int FusionPass::localApply(LinkedList* list) {
    return 0;
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

int PrecisionPass::localApply(LinkedList* list) {
    return 0;
};

int QuantizationPass::globalApply(LinkedList* list) {
    Node* current = list->head;

    // add in quantization node here
    do
    while (current != nullptr) {

    }

    // add in another one in the end
};

int QuantizationPass::localApply(LinkedList* list) {

};


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

int BackendPass::localApply(LinkedList* list) {
    return 0;
};

void PassManager::registerPass(Pass* pass) {
    passes.push_back(pass);
};

// run global runs full optimization one by one
void PassManager::runGlobal() {
    for (Pass* pass : passes) {
       pass->globalApply(linkedList);
    }
};

// TODO run local will run the local optimizations for all the passes
// one sequential pass instead of multiple
// fusion collapses nodes so it's harder
void PassManager::runLocal() {
    return;
};

bool PassManager::verify() {
    return false;
};
