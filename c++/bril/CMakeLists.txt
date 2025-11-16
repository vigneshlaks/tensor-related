cmake_minimum_required(VERSION 3.10)

# high level optimization
project(hlo)

set(CMAKE_CXX_STANDARD 17)

find_package(nlohmann_json)
add_library(cfg cfg.cpp cfg.h)
add_library(lvn lvn.cpp lvn.h)
add_executable(tdce tdce.cpp tdce.h)

target_link_libraries(cfg nlohmann_json::nlohmann_json)
target_link_libraries(tdce nlohmann_json::nlohmann_json cfg)
target_link_libraries(lvn nlohmann_json::nlohmann_json)