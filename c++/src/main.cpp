#include <iostream>
#include "../include/types.h"
#include "../include/ops.h"
#include "../include/frontend.h"
#include "../include/passes.h"
#include <fstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

int main() {
        std::ifstream file("./ir/1_layer.json");
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
                current->operation->execute();
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


        return 0;
}
