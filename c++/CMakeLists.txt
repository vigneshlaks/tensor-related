cmake_minimum_required(VERSION 3.10)

project(ttc)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(nlohmann_json REQUIRED)

set(CMAKE_C_COMPILER /opt/homebrew/opt/llvm/bin/clang)
set(CMAKE_CXX_COMPILER /opt/homebrew/opt/llvm/bin/clang++)

add_library(frontend frontend.cpp frontend.h)
add_library(ops ops.cpp ops.h)
add_library(types types.cpp types.h)
add_library(passes passes.cpp passes.h)

target_link_libraries(frontend nlohmann_json::nlohmann_json)
target_link_libraries(passes nlohmann_json::nlohmann_json)