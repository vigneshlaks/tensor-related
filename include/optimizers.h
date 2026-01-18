#ifndef OPTIMIZERS_H
#define OPTIMIZERS_H

#include "hlo.h"
#include <vector>
#include <cmath>
#include <unordered_map>
#include <string>

class Optimizers {
protected:
    float learningRate;
    LinkedList* list;

public:
    Optimizers(float lr, LinkedList* l) : learningRate(lr), list(l) {}

    void forward();
    void backward();
    void zeroGrad();

    virtual void descentStep() = 0;
    virtual ~Optimizers() = default;
};

class SGD : public Optimizers {
public:
    SGD(float lr, LinkedList* l) : Optimizers(lr, l) {}
    void descentStep() override;
};

class Adam : public Optimizers {
private:
    float beta1;
    float beta2;
    float epsilon;
    int t = 0;

    // ID to momentum / velocity vectors
    std::unordered_map<std::string, std::vector<float>> m;
    std::unordered_map<std::string, std::vector<float>> v;

public:
    Adam(float lr, LinkedList* l, float b1, float b2, float e);
    void descentStep() override;
};

#endif
