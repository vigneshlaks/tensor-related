#include <iostream>
#include "../include/types.h"
#include "../include/ops.h"
#include "../include/frontend.h"
#include "../include/passes.h"
#include <fstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

int main(int argc, char *argv[])
{
    ComputeGraph graph;
    std::ifstream file("./ir/1_layer.json");
    std::vector<Pass*> passes;
    PassManager pm(&graph, passes);

    BackendPass bp(Backend::CPU);
    pm.registerPass(&bp);

    return 0;
}
