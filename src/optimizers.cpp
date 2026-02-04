#include <iostream>
#include <algorithm>
#include "../include/optimizers.h"
#include "../include/ops.h"
// everything related to parameters

// might have to change params for other inputs
// places inputs before computation
void Optimizers::forward(std::vector<uint8_t> input, uint8_t output) {
    Node *current = list->head;

    while (current != nullptr && !current->id.empty()) {
        if (dynamic_cast<ConstOp*>(current->operation.get()) != nullptr && !current->trainable) {
            // fill Tensor with normalized input
            float maxVal = *std::max_element(input.begin(), input.end());
            for (int i = 0; i < current->output->storage.size(); i++) {
                current->output->storage[i] = (maxVal > 0) ? input[i] / maxVal : 0.0f;
            }
        } else if (LossOp* lossOp = dynamic_cast<LossOp*>(current->operation.get())) {
            // one hot encode
            for (size_t i = 0; i < lossOp->groundTruth->storage.size(); i++) {
                lossOp->groundTruth->storage[i] = 0.0f;
            }
            lossOp->groundTruth->storage[output] = 1.0f;
        }
        
        // calling computation
        if (current->operation != nullptr) {
            current->operation->forward();
        }

        current = current->next;
    }
};

void Optimizers::backward() {
    Node *current = list->tail;
    list->tail->output->grad[0] = 1.0f;

    while (current != nullptr) {
        if (current->operation != nullptr) {
            current->operation->backward();
        }
        current = current->prev;
    }
};

void Optimizers::zeroGrad() {
    Node *current = list->head;

    while (current != nullptr && !current->id.empty()) {
        if (current->output != nullptr) {
            for (size_t i = 0; i < current->output->grad.size(); i++) {
                current->output->grad[i] = 0.0f;
            }
        }
        current = current->next;
    }
};

void SGD::descentStep() {
    Node *current = list->head;

    while (current != nullptr && !current->id.empty())
    {
        // check if operation is trainable
        // really only matmul
        if (current->trainable && current->output != nullptr)
        {
            // update outputs
            for (size_t i = 0; i < current->output->storage.size(); i++)
            {
                current->output->storage[i] -= learningRate * current->output->grad[i];
            }
        }
        current = current->next;
    }
};

Adam::Adam(float lr, LinkedList *l, float b1, float b2, float e) : Optimizers(lr, l) {
    beta1 = b1;
    beta2 = b2;
    epsilon = e;

    Node *current = list->head;

    while (current != nullptr && !current->id.empty())
    {
        if (current->trainable && current->output != nullptr)
        {
            size_t size = current->output->storage.size();
            m[current->id] = std::vector<float>(size, 0.0f);
            v[current->id] = std::vector<float>(size, 0.0f);
        }
        current = current->next;
    }
}

void Adam::descentStep() {
    t++;
    Node *current = list->head;

    // null or boundary node
    while (current != nullptr && !current->id.empty())
    {
        // check if operation is trainable
        // really only matmul
        if (current->trainable && current->output != nullptr) {
            for (size_t i = 0; i < current->output->storage.size(); i++) {
                // Adam update formula
                float g = current->output->grad[i];

                m[current->id][i] = beta1 * m[current->id][i] + (1.0f - beta1) * g;
                v[current->id][i] = beta2 * v[current->id][i] + (1.0f - beta2) * g * g;

                float m_hat = m[current->id][i] / (1.0f - std::pow(beta1, t));
                float v_hat = v[current->id][i] / (1.0f - std::pow(beta2, t));

                current->output->storage[i] -= learningRate * m_hat / (std::sqrt(v_hat) + epsilon);
            }
        }
        current = current->next;
    }
}
