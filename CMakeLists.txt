cmake_minimum_required(VERSION 3.10)
project(tensor)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(tensor 
    main.cpp 
    ops.cpp 
    types.cpp
)