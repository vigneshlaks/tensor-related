#include <iostream>
#include "../include/types.h"
#include "../include/ops.h"
#include "../include/hlo.h"
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
}

int main(int argc, char* argv[]) {
    test_ir("irs/two_dimensional/1_layer.json");
    return 0;
}
