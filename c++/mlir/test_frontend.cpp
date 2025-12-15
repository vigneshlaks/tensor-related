#include "frontend.h"
#include <iostream>
#include <cassert>

void test_single_const() {
    json instrs = json::array({
        {{"id", "x"}, {"op", "const"}, {"value", std::vector<float>{1.0, 2.0, 3.0, 4.0}}, {"dim", std::vector<size_t>{2, 2}}}
    });

    ComputeGraph cg = parseIR(instrs);
    assert(cg.nodeMap.count("x") == 1);
    assert(cg.nodeMap["x"]->output->dimension[0] == 2);
    assert(cg.nodeMap["x"]->output->dimension[1] == 2);
    std::cout << "✓ test_single_const passed\n";
}

void test_matmul_chain() {
    json instrs = json::array({
        {{"id", "w"}, {"op", "const"}, {"value", std::vector<float>(6, 1.0)}, {"dim", std::vector<size_t>{2, 3}}},
        {{"id", "x"}, {"op", "const"}, {"value", std::vector<float>(12, 1.0)}, {"dim", std::vector<size_t>{3, 4}}},
        {{"id", "y"}, {"op", "matmul"}, {"args", json::array({"w", "x"})}}
    });

    ComputeGraph cg = parseIR(instrs);
    assert(cg.nodeMap["y"]->output->dimension[0] == 2);
    assert(cg.nodeMap["y"]->output->dimension[1] == 4);
    assert(cg.nodeMap["y"]->operation != nullptr);
    assert(cg.nodeMap["y"]->operation->verify());
    std::cout << "✓ test_matmul_chain passed\n";
}

void test_matmul_relu() {
    json instrs = json::array({
        {{"id", "w"}, {"op", "const"}, {"value", std::vector<float>(4, 1.0)}, {"dim", std::vector<size_t>{2, 2}}},
        {{"id", "x"}, {"op", "const"}, {"value", std::vector<float>(4, 1.0)}, {"dim", std::vector<size_t>{2, 2}}},
        {{"id", "mm"}, {"op", "matmul"}, {"args", json::array({"w", "x"})}},
        {{"id", "out"}, {"op", "relu"}, {"args", json::array({"mm"})}}
    });

    ComputeGraph cg = parseIR(instrs);
    assert(cg.nodeMap["out"]->operation != nullptr);
    assert(cg.nodeMap["out"]->operation->verify());
    std::cout << "✓ test_matmul_relu passed\n";
}

void test_full_pipeline() {
    json instrs = json::array({
        {{"id", "w1"}, {"op", "const"}, {"value", std::vector<float>(4, 0.5)}, {"dim", std::vector<size_t>{2, 2}}},
        {{"id", "x"}, {"op", "const"}, {"value", std::vector<float>(4, 1.0)}, {"dim", std::vector<size_t>{2, 2}}},
        {{"id", "mm1"}, {"op", "matmul"}, {"args", json::array({"w1", "x"})}},
        {{"id", "a1"}, {"op", "relu"}, {"args", json::array({"mm1"})}},
        {{"id", "target"}, {"op", "const"}, {"value", std::vector<float>(4, 1.0)}, {"dim", std::vector<size_t>{2, 2}}},
        {{"id", "loss"}, {"op", "mse_loss"}, {"args", json::array({"a1"})}}
    });

    ComputeGraph cg = parseIR(instrs);
    assert(cg.nodeMap.size() == 6);
    assert(cg.nodeMap["loss"]->operation != nullptr);
    std::cout << "✓ test_full_pipeline passed\n";
}

void test_linked_list_structure() {
    json instrs = json::array({
        {{"id", "a"}, {"op", "const"}, {"value", std::vector<float>(2, 1.0)}, {"dim", std::vector<size_t>{1, 2}}},
        {{"id", "b"}, {"op", "const"}, {"value", std::vector<float>(2, 1.0)}, {"dim", std::vector<size_t>{1, 2}}},
        {{"id", "c"}, {"op", "const"}, {"value", std::vector<float>(2, 1.0)}, {"dim", std::vector<size_t>{1, 2}}}
    });

    ComputeGraph cg = parseIR(instrs);

    // Verify linked list connections
    Node* curr = cg.nodeMap["a"];
    assert(curr->next != nullptr);
    assert(curr->next->id == "b");
    assert(curr->next->next->id == "c");

    std::cout << "✓ test_linked_list_structure passed\n";
}

int main() {
    std::cout << "Running frontend tests...\n\n";

    test_single_const();
    test_matmul_chain();
    test_matmul_relu();
    test_full_pipeline();
    test_linked_list_structure();

    std::cout << "\n✓ All tests passed!\n";
    return 0;
}
