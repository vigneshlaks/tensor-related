#include "../include/optimizers.h"


//TODO Add helper for ll traversals


void Optimizers::forward() {
    Node* current = list->head;

    while (current != nullptr && !current->id.empty()) {
        if (current->operation != nullptr) {
            current->operation->forward();
        }
        current = current->next;
    }
}

void Optimizers::backward() {
    Node* current = list->tail;

    while (current != nullptr) {
        if (current->operation != nullptr) {
            current->operation->backward();
        }
        current = current->prev;
    }
}

void Optimizers::zeroGrad() {
    Node* current = list->head;

    while (current != nullptr && !current->id.empty()) {
        if (current->output != nullptr) {
            for (size_t i = 0; i < current->output->grad.size(); i++) {
                current->output->grad[i] = 0.0f;
            }
        }
        current = current->next;
    }
}

void SGD::descentStep() {
    Node* current = list->head;

    while (current != nullptr && !current->id.empty()) {
        if (current->trainable && current->output != nullptr) {
            for (size_t i = 0; i < current->output->storage.size(); i++) {
                current->output->storage[i] -= learningRate * current->output->grad[i];
            }
        }
        current = current->next;
    }
}

Adam::Adam(float lr, LinkedList* l, float b1, float b2, float e) : Optimizers(lr, l) {
    beta1 = b1;
    beta2 = b2;
    epsilon = e;

    Node* current = list->head;

    while (current != nullptr && !current->id.empty()) {
        if (current->trainable && current->output != nullptr) {
            size_t size = current->output->storage.size();
            m[current->id] = std::vector<float>(size, 0.0f);
            v[current->id] = std::vector<float>(size, 0.0f);
        }
        current = current->next;
    }
}

void Adam::descentStep() {
    t++;
    Node* current = list->head;

    while (current != nullptr && !current->id.empty()) {
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
