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

        ComputeGraph graph = parseJSON(inputIR);
        std::cout << "Before optimization:" << std::endl;
        printComputeGraph(graph);

        std::vector<Pass*> passes;
        PassManager pm(&graph, passes);

        BackendPass bp(Backend::CPU);
        pm.registerPass(&bp);

        // FusionPass fp;
        // pm.registerPass(&fp);

        // Precision p = Int8;
        // QuantizationPass qp(p);
        // pm.registerPass(&qp);
        
        pm.runGlobal();

        std::cout << "\nAfter optimization:" << std::endl;
        printComputeGraph(graph);

        Node* current = graph.head;
        while (current != nullptr) {
            if (current->operation != nullptr) {
                current->operation->execute();
            }
            current = current->next;
        }

        std::cout << "\nExecution complete!" << std::endl;

        std::cout << "\nFinal Results:" << std::endl;
            
        current = graph.head;
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
