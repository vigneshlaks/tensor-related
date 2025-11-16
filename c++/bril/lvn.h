#include <nlohmann/json.hpp>

using json = nlohmann::json;

// Local Value Numbering optimizations
// These functions apply compiler optimizations to a single basic block

// Eliminate redundant subexpressions through constant propagation
std::vector<json> constant_folding(std::vector<json> block);

// Replace variables with their known constant values
std::vector<json> copy_propagation(std::vector<json> block);

// Eliminate common subexpressions through value numbering
std::vector<json> cse(std::vector<json> block);