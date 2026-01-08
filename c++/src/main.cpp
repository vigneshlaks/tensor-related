#include <iostream>
#include "../include/types.h"
#include "../include/ops.h"
#include "../include/frontend.h"
#include "../include/passes.h"
#include <fstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

void test_ir(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cout << "Failed to open " << filename << std::endl;
            return;
        }
        json inputIR;
        file >> inputIR;

        Metadata meta = parseMetaData(inputIR);
        LinkedList list = parseJSON(inputIR);

        std::cout << "Before optimization:" << std::endl;
        printLinkedList(list);

        PassManager pm(&list, meta.passes);
        pm.runGlobal();

        std::cout << "\nAfter optimization:" << std::endl;
        printLinkedList(list);

        Node* current = list.head;
        while (current != nullptr) {
            std::cout << "Executing: " << current->id << std::endl;
            if (current->operation != nullptr) {
                current->operation->forward();
            }
            current = current->next;
        }

        std::cout << "\nExecution complete!" << std::endl;

        std::cout << "\nFinal Results:" << std::endl;

        current = list.head;
        while (current != nullptr) {
            if (current->output != nullptr) {
                std::cout << current->id << ": ";
                for (float val : current->output->storage) {
                    std::cout << val << " ";
                }
                std::cout << std::endl;
            }
            current = current->next;
        }

        float learning_rate = 0.0001f;
        int epochs = 500;

        for (int epoch = 0; epoch < epochs; epoch++) {
            current = list.head;
            while (current != nullptr) {
                if (current->output != nullptr) {
                    for (size_t i = 0; i < current->output->grad.size(); i++) {
                        current->output->grad[i] = 0.0f;
                    }
                }
                current = current->next;
            }

            current = list.head;
            while (current != nullptr) {
                if (current->opType != OpType::Const && current->output != nullptr) {
                    for (size_t i = 0; i < current->output->storage.size(); i++) {
                        current->output->storage[i] = 0.0f;
                    }
                }
                current = current->next;
            }

            current = list.head;
            while (current != nullptr) {
                if (current->operation != nullptr) {
                    current->operation->forward();
                }
                current = current->next;
            }

            if (list.tail->output != nullptr) {
                for (size_t i = 0; i < list.tail->output->grad.size(); i++) {
                    list.tail->output->grad[i] = 1.0f;
                }
            }

            current = list.tail;
            while (current != nullptr) {
                if (current->operation != nullptr) {
                    current->operation->backward();
                }
                current = current->prev;
            }

            current = list.head;
            while (current != nullptr) {
                if (current->opType == OpType::Const && current->output != nullptr) {
                    bool has_gradient = false;
                    for (float grad : current->output->grad) {
                        if (grad != 0.0f) {
                            has_gradient = true;
                            break;
                        }
                    }

                    if (has_gradient && current->id != "input") {
                        for (size_t i = 0; i < current->output->storage.size(); i++) {
                            current->output->storage[i] -= learning_rate * current->output->grad[i];
                        }
                    }
                }
                current = current->next;
            }

            if (epoch % 50 == 0 || epoch == epochs - 1) {
                float total_loss = 0.0f;
                if (list.tail->output != nullptr) {
                    for (float val : list.tail->output->storage) {
                        total_loss += val;
                    }
                }
                std::cout << "Epoch " << epoch << " - Loss: " << total_loss;

                Node* output_node = list.tail->prev;
                if (output_node && output_node->output) {
                    std::cout << " - Output: ";
                    for (size_t i = 0; i < std::min((size_t)4, output_node->output->storage.size()); i++) {
                        std::cout << output_node->output->storage[i] << " ";
                    }
                    if (output_node->output->storage.size() > 4) std::cout << "...";
                }
                std::cout << std::endl;
            }
        }

        std::cout << "\nFinal trained weights:" << std::endl;
        current = list.head;
        while (current != nullptr) {
            if (current->opType == OpType::Const && current->id != "input" && current->output != nullptr) {
                std::cout << current->id << ": ";
                for (size_t i = 0; i < std::min((size_t)8, current->output->storage.size()); i++) {
                    std::cout << current->output->storage[i] << " ";
                }
                if (current->output->storage.size() > 8) std::cout << "...";
                std::cout << std::endl;
            }
            current = current->next;
        }

}

int main() {
        test_ir("irs/two_dimensional/1_layer.json");
        std::cout << "\n\n";
        test_ir("irs/two_dimensional/no_quantization.json");
        std::cout << "\n\n";
        test_ir("irs/two_dimensional/large_matrix.json");
        return 0;
}
