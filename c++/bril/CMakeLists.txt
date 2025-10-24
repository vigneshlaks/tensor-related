cmake_minimum_required(VERSION 3.10)
project(cfg)

set(CMAKE_CXX_STANDARD 17)

find_package(nlohmann_json 3.2.0 REQUIRED)

add_executable(program cfg.cpp)
target_link_libraries(program nlohmann_json::nlohmann_json)