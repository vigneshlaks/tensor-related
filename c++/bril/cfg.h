#include <vector>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

std::vector<std::vector<json>> form_blocks(json instrs);

void print_blocks(std::vector<std::vector<json>> blocks);