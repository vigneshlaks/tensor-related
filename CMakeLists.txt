cmake_minimum_required(VERSION 3.18)

project(ttc)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(nlohmann_json REQUIRED)
find_package(CUDA)

if (CUDA_FOUND)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    # for older compilers and cmake
    # set(CMAKE_CUDA_ARCHITECTURES 86)
    message("Building with GPU support")
    add_library(gpu_exec src/gpu_exec.cu include/gpu_exec.h)
    target_link_libraries(gpu_exec PRIVATE cublas)
    set(OPS_GPU_LIB gpu_exec)
    add_compile_definitions(CUDA_FOUND) 
else()
    message("Building with CPU support")
    set(OPS_GPU_LIB "")
endif()

add_library(frontend src/frontend.cpp include/frontend.h)
add_library(optimizers src/optimizers.cpp include/optimizers.h)
add_library(ops src/ops.cpp include/ops.h)
add_library(types src/types.cpp include/types.h)
add_library(passes src/passes.cpp include/passes.h)

add_executable(main src/main.cpp)

target_link_libraries(frontend nlohmann_json::nlohmann_json)
target_link_libraries(passes nlohmann_json::nlohmann_json)
target_link_libraries(optimizers nlohmann_json::nlohmann_json)
target_link_libraries(main optimizers frontend types ops passes ${OPS_GPU_LIB} nlohmann_json::nlohmann_json)