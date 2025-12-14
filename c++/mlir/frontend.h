// go from json to ir for now
#ifndef FRONTEND_H
#define FRONTEND_H

#include "ops.h"
#include "types.h"

#include <nlohmann/json.hpp>
using json = nlohmann::json;

class IRBuilder {
    std::vector<std::pair<Tensor, Op>> parseIR(json inputIR);
};

#endif