#include <iostream>
#include "../include/types.h"
#include "../include/ops.h"
#include "../include/frontend.h"
#include "../include/passes.h"
#include "../include/optimizers.h"
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

        int numEpochs = 5;
        // learning rate
        SGD sgd = SGD(0.01f, &list);

        std::cout << "\nTraining:" << std::endl;
        for (int i = 0; i < numEpochs; i++) {
            sgd.forward();

            // print loss
            float loss = list.tail->output->storage[0];
            std::cout << "Epoch " << i << " | Loss: " << loss << std::endl;

            // sets loss of tail node to 1
            sgd.init();
            sgd.backward();
            sgd.descentStep();
            sgd.zeroGrad();
        }

        std::cout << "\nExecution complete!" << std::endl;
};

int main(int argc, char* argv[]) {
    test_ir("irs/two_dimensional/test.json");
    return 0;
}
