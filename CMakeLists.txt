cmake_minimum_required(VERSION 3.18)

project(ttc)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(nlohmann_json REQUIRED)
find_package(CUDA)

if (CUDA_FOUND)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    message("Building with GPU support")
    add_library(gpu_exec src/gpu_exec.cu include/gpu_exec.h)
    target_link_libraries(gpu_exec PRIVATE cuda)
    set(OPS_GPU_LIB gpu_exec)
else()
    message("Building with CPU support")
    set(OPS_GPU_LIB "")
endif()

add_library(frontend src/frontend.cpp include/frontend.h)
add_library(ops src/ops.cpp include/ops.h)
add_library(types src/types.cpp include/types.h)
add_library(passes src/passes.cpp include/passes.h)

add_executable(main src/main.cpp)

target_link_libraries(frontend nlohmann_json::nlohmann_json)
target_link_libraries(passes nlohmann_json::nlohmann_json)
target_link_libraries(main frontend types ops passes ${OPS_GPU_LIB} nlohmann_json::nlohmann_json)

# Testing with Catch2
# find_package(Catch2 3 REQUIRED)

# enable_testing()

# add_executable(test_types tests/test_types.cpp)
# target_link_libraries(test_types types Catch2::Catch2WithMain)

# include(CTest)
# include(Catch)
# catch_discover_tests(test_types)