cmake_minimum_required(VERSION 3.10)

project(ttc)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(nlohmann_json REQUIRED)
set(CMAKE_CXX_COMPILER /usr/bin/clang++)

add_library(frontend frontend.cpp frontend.h)
add_library(ops ops.cpp ops.h)
add_library(types types.cpp types.h)
add_library(passes passes.cpp passes.h)

add_executable(main main.cpp)

target_link_libraries(frontend nlohmann_json::nlohmann_json)
target_link_libraries(passes nlohmann_json::nlohmann_json)
target_link_libraries(main frontend types ops passes nlohmann_json::nlohmann_json)