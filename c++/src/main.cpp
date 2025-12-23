#include <iostream>
#include "types.h"
#include "ops.h"
#include "frontend.h"
#include "passes.h"
#include <fstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

int main(int argc, char* argv[]) {
    ComputeGraph graph;
    std::ifstream file("1_layer.json");
    if (!file.is_open()) {
        std::cerr << "Error: Could not open lay.json" << std::endl;
        return 1;
    }

    ComputeGraph cGraph = parseJSON(json::parse(file));
    printComputeGraph(cGraph);

    return 0;
}
