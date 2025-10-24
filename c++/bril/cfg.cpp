#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

struct Block {
    int id;
    double value;
};

int main() {
    std::ifstream f("example.json");
    json data = json::parse(f);
    
    // get funct
    auto instrs = data["functions"][0]["instrs"];
        
    // Iterate through each instruction
    for (auto& instr : instrs) {
        std::cout << instr.dump(2) << "\n";
        std::cout << std::endl;
    }

    return 0;
}