// left as 3 different optimizations for now
#include "lvn.h"
#include <unordered_map>
#include <string>
#include <tuple>
#include <algorithm>

using ValueType = std::variant<std::tuple<std::string, std::vector<int>>, int>;
using TableType = std::unordered_map<std::string, std::tuple<int, ValueType, std::string>>;

namespace {
    ValueType canonicalize(json& instr, TableType& var2num) {
        // get operation and lvn numbers
        std::string op = instr["op"].get<std::string>();
        std::vector<int> nums;

        // get lvn num for each variable in the arguments
        for (const auto& arg : instr["args"]) {
            std::string argVar = arg.get<std::string>();
            if (var2num.find(argVar) != var2num.end()) {
                int lvnNum = std::get<0>(var2num[argVar]);
                nums.push_back(lvnNum);
            }
        }

        // sort for canonical representation
        std::sort(nums.begin(), nums.end());

        return std::make_tuple(op, nums);
    }
}

std::vector<json> constant_folding(std::vector<json> block) {
    // number here refers to table row
    TableType var2num;  // variable to (lvn, value, canonical_var)
    std::unordered_map<std::string, ValueType> val2var;  // value to variable
    int lvnCount = 0;
    std::vector<json> result;

    for (json instr : block) {
        std::string var = instr["dest"].get<std::string>();
        ValueType value;

        if (instr.contains("args")) {
            // arithmetic op
            value = canonicalize(instr, var2num);
        } else if (instr.contains("value")) {
            // const
            value = instr["value"].get<int>();
        } else {
            // non-arithmetic op
            result.push_back(instr);
            continue;
        }

        // value type to string
        std::string valueStr = value.index() == 0 ? std::get<std::string>(std::get<0>(value)) : std::to_string(std::get<int>(value));

        // check if we've seen this value before
        if (val2var.find(valueStr) != val2var.end()) {
            // value computed
            ValueType canonicalValue = val2var[valueStr];
            std::string canonicalVar;

            if (canonicalValue.index() == 1) {
                // some constant
                instr["value"] = std::get<int>(canonicalValue);
                instr.erase("args");
            } else {
                auto it = var2num.begin();
                while (it != var2num.end()) {
                    if (std::get<1>(it->second) == canonicalValue) {
                        canonicalVar = std::get<2>(it->second);
                        break;
                    }
                    ++it;
                }
                instr["op"] = "id";
                instr["args"] = {canonicalVar};
            }
        } else {
            val2var[valueStr] = value;
        }

        var2num[var] = std::make_tuple(lvnCount, value, var);
        lvnCount++;

        result.push_back(instr);
    }

    return result;
}

// propagate constant values and replace variables with constant copies
std::vector<json> copy_propagation(std::vector<json> block) {
    std::unordered_map<std::string, int> var2const;
    std::vector<json> result;

    for (json instr : block) {
        // constant expressions
        if (instr.contains("value") && instr.contains("dest")) {
            std::string var = instr["dest"].get<std::string>();
            int val = instr["value"].get<int>();
            var2const[var] = val;
        }

        if (instr.contains("args")) {
            auto args = instr["args"].get<std::vector<std::string>>();
            for (auto& arg : args) {
                // replace if we find it
                if (var2const.find(arg) != var2const.end()) {
                    arg = std::to_string(var2const[arg]);
                }
            }
            instr["args"] = args;
        }

        result.push_back(instr);
    }

    return result;
}

// cse - remove duplicate computations
std::vector<json> cse(std::vector<json> block) {
    std::unordered_map<std::string, std::string> expr2var;
    std::vector<json> result;

    for (json instr : block) {
        if (!instr.contains("op") || !instr.contains("dest")) {
            result.push_back(instr);
            continue;
        }

        std::string op = instr["op"].get<std::string>();
        std::string dest = instr["dest"].get<std::string>();

        // represent operation
        std::string expr = op;
        if (instr.contains("args")) {
            auto args = instr["args"].get<std::vector<std::string>>();
            std::sort(args.begin(), args.end());
            for (const auto& arg : args) {
                expr += "_" + arg;
            }
        }

        if (expr2var.find(expr) != expr2var.end()) {
            std::string canonicalVar = expr2var[expr];
            json copyInstr;
            copyInstr["op"] = "id";
            copyInstr["dest"] = dest;
            copyInstr["args"] = {canonicalVar};
            
            // push result back
            result.push_back(copyInstr);
        } else {
            // save new operation
            expr2var[expr] = dest;
            result.push_back(instr);
        }
    }

    return result;
}