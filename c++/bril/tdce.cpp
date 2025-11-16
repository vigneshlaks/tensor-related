#include <fstream>
#include <iostream>
#include <unordered_set>
#include "cfg.h"

std::vector<json> tdce(std::vector<json> block) {
    std::vector<json> newBlock;
    bool changed = true;

    while (changed) {
        // initialize a set of values to delete
        std::unordered_set<std::string> delSet;

        for (json& instr : block) {
            if (instr.contains("args")) {
                for (std::string arg : instr["args"]) {
                    if (delSet.count(arg) > 0) {
                        delSet.erase(arg);
                    }
                }
            }
            if (instr.contains("dest")) {
                delSet.insert(instr["dest"]);
            }
        }

        if (delSet.size() == 0) {
            changed = false;
        } else {
            // create newBlock with values not from current block
            for (json& instr : block) {
                bool keep = false;
                
                if (instr.contains("dest")) {
                    // it it's not in the delset then keep it
                    keep = delSet.find(instr["dest"]) == delSet.end();
                } else{
                    keep = true;
                }

                if (keep) {
                    newBlock.push_back(instr);
                }
            }
            block = newBlock;
            newBlock.clear();
        }
    }

    return block;
}

int main() {
    std::ifstream f = std::ifstream("combo.json");

    if (!f.is_open()) {
        std::cerr << "Error opening file" << std::endl;
        return 1;
    }

    json data = json::parse(f);
    auto instrs = data["functions"][0]["instrs"];
 
    std::vector<std::vector<json>> blocks = form_blocks(instrs);

    for (int i = 0; i < blocks.size(); i++) {
        blocks[i] = tdce(blocks[i]);
    }

    print_blocks(blocks);

    return 0;
}