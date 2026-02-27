#include <iostream>
#include "../include/types.h"
#include "../include/ops.h"
#include "../include/frontend.h"
#include "../include/passes.h"
#include "../include/optimizers.h"
#include <fstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

struct MNISTData {
    // vector representing images (inputs)
    std::vector<std::vector<uint8_t>> images;
    // labels representing outputs
    std::vector<uint8_t> labels;
};

// 32 bit int
uint32_t readInt(std::ifstream& file) {
    uint8_t bytes[4];
    // read 4 bytes
    // the char casting to
    // work with file.read
    file.read((char*) bytes, 4);

    // shift bits over and combine
    return (bytes[0] << 24) |
           (bytes[1] << 16) |
           (bytes[2] << 8)  |
           (bytes[3]);
}

auto loadData(std::string inputs, std::string outputs) {
    // make into tensors
    // do batching
    MNISTData data;
    std::ifstream imgFile(inputs, std::ios::binary);
    if (!imgFile.is_open()) {
        std::cerr << "Failed to open " << inputs << std::endl;
    }

    // the initial info from the file
    // is a series of integers
    uint32_t imgMagic = readInt(imgFile);
    uint32_t numImages = readInt(imgFile);
    uint32_t numRows = readInt(imgFile);
    uint32_t numCols = readInt(imgFile);

    data.images.resize(numImages);
    for (uint32_t i = 0; i < numImages; i++) {
        data.images[i].resize(numRows * numCols);
        // the char casting to  
        // work with file.read
        // .data() get the memory location
        // imgFile.read and place into the .data memory location
        imgFile.read((char*)data.images[i].data(), numRows * numCols);
    }

    std::ifstream lblFile(outputs, std::ios::binary);
    if (!lblFile.is_open()) {
        std::cerr << "Failed to open " << outputs << std::endl;
        return data;
    }

    // similar for labels
    uint32_t lblMagic = readInt(lblFile);
    uint32_t numLabels = readInt(lblFile);

    data.labels.resize(numLabels);
    lblFile.read((char*)data.labels.data(), numLabels);
    lblFile.close();

    std::cout << "Loaded " << data.images.size() << " images and " 
              << data.labels.size() << " labels\n";
    
    return data;
}

int main(int argc, char* argv[]) {
    auto trainData = loadData("data/MNIST/raw/train-images-idx3-ubyte",
                               "data/MNIST/raw/train-labels-idx1-ubyte");
    auto testData = loadData("data/MNIST/raw/t10k-images-idx3-ubyte",
                              "data/MNIST/raw/t10k-labels-idx1-ubyte");

    std::cout << "Train images: " << trainData.images.size() << std::endl;
    std::cout << "Train labels: " << trainData.labels.size() << std::endl;

    std::string filename = "irs/mnist/mnist.json";

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cout << "Failed to open " << filename << std::endl;
        return 0;
    }
    json inputIR;
    file >> inputIR;

    Metadata meta = parseMetaData(inputIR);
    LinkedList list = parseJSON(inputIR);

    PassManager pm(&list, meta.passes);
    pm.runGlobal();

    SGD sgd = SGD(0.01f, &list);
    int numEpochs = 3;
    size_t numSamples = 1000;

    #ifdef CUDA_FOUND
    std::cout << "Running on GPU" << std::endl;
    BackendPass gpuPass(GPU);
    gpuPass.globalApply(&list);
    sgd.initDevice();
    #endif

    #ifdef METAL_FOUND
    std::cout << "Running on Metal (M1)" << std::endl;
    BackendPass metalPass(METAL);
    metalPass.globalApply(&list);
    sgd.initDevice();
    #endif

    #if !defined(CUDA_FOUND) && !defined(METAL_FOUND)
    std::cout << "Running on CPU" << std::endl;
    #endif

    

    std::cout << "\n--- Training ---" << std::endl;
    for (int epoch = 0; epoch < numEpochs; epoch++) {
        float epochLoss = 0.0f;
        for (size_t i = 0; i < numSamples; i++) {
            sgd.zeroGrad();
            sgd.forward(trainData.images[i], trainData.labels[i]);

            list.tail->output->toHost();
            epochLoss += list.tail->output->storage[0];

            sgd.backward();
            sgd.descentStep();
        }

        std::cout << "Epoch " << epoch << " | Avg Loss: " << epochLoss / numSamples << std::endl;
    }

    sgd.syncToHost();

    return 0;
}