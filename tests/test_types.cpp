#include <catch2/catch_test_macros.hpp>
#include "../include/types.h"

TEST_CASE("Tensor construction with dimensions", "[tensor]") {
    Tensor t({2, 3});
    REQUIRE(t.dimension.size() == 2);
    REQUIRE(t.storage.size() == 6);
    REQUIRE(t.grad.size() == 6);
}

TEST_CASE("Tensor construction with storage", "[tensor]") {
    Tensor t({2, 3}, {1, 2, 3, 4, 5, 6});
    REQUIRE(t.getValue({0, 0}) == 1);
    REQUIRE(t.getValue({1, 2}) == 6);
}

TEST_CASE("Invalid storage size throws", "[tensor]") {
    REQUIRE_THROWS_AS(Tensor({2, 3}, {1, 2, 3}), std::invalid_argument);
}

TEST_CASE("Stride calculation", "[tensor]") {
    Tensor t({2, 3, 4});
    REQUIRE(t.stride[0] == 12);
    REQUIRE(t.stride[1] == 4);
    REQUIRE(t.stride[2] == 1);
}

TEST_CASE("Set and get value", "[tensor]") {
    Tensor t({2, 2});
    t.setValue({1, 0}, 5.0f);
    REQUIRE(t.getValue({1, 0}) == 5.0f);
}

TEST_CASE("Gradient operations", "[tensor]") {
    Tensor t({2, 2});
    t.setGrad({0, 0}, 1.0f);
    t.accumulateGrad({0, 0}, 2.0f);
    REQUIRE(t.getGrad({0, 0}) == 3.0f);
}

TEST_CASE("Out of bounds throws", "[tensor]") {
    Tensor t({2, 3});
    REQUIRE_THROWS_AS(t.getValue({5, 0}), std::invalid_argument);
    REQUIRE_THROWS_AS(t.setValue({0, 10}, 1.0f), std::invalid_argument);
}
