#include "cfg.h"
#include <iostream>
#include <fstream>
#include <map>
#include <algorithm>

namespace {
    std::vector<std::string> TERMINATORS{"br", "jmp", "ret"};
};

std::vector<std::vector<json>> form_blocks(json instrs) {
    std::vector<std::vector<json>> blocks;
    std::vector<json> curr_block;

    for (json instr : instrs) {
        // if it's an op instruction
        if (instr.contains("op")) {
            curr_block.push_back(instr);

            // check if it's a terminating operation
            if (std::find(TERMINATORS.begin(), TERMINATORS.end(), instr["op"]) != TERMINATORS.end()) {
                blocks.push_back(curr_block);
                curr_block.clear();
            }
        }
        // if not it's a label instruction
        else {
            // add current block if it's non empty
            if (curr_block.size() > 0) {
                blocks.push_back(curr_block);
                curr_block.clear();
            }
            curr_block.push_back(instr);
        }
    }

    // check for final block then push back
    if (curr_block.size() > 0) {
        blocks.push_back(curr_block);
        curr_block.clear();
    }

    return blocks;
}

void print_blocks(std::vector<std::vector<json>> blocks) {
    for (std::vector<json>& block : blocks) {
        if (block[0].contains("op")) {
            std::cout << "anon block" << std::endl;
            for (json& instr : block) {
                std::cout << instr << std::endl;
            }
            // to denote end of block
            std::cout << std::endl << std::endl;
        } else {
            // print name of label for start of block
            std::cout << block[0]["label"] << std::endl;
            for (int i = 1; i < blocks.size(); i++) {
                std::cout << blocks[i] << std::endl;
            }
            // to denote end of block
            std::cout << std::endl << std::endl;
        }
    }
}