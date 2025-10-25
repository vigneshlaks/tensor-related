#ifndef FRONTEND_H
#define FRONTEND_H

#include <string>
#include "ops.h"

class SSABuilder {
private:
    int opNum = 0;

public:
    std::string buildOp (Op& op);
};

#endif