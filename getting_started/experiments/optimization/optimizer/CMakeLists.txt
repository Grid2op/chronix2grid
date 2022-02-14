cmake_minimum_required(VERSION 3.22)
project(Optimizer)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -pthread")

include_directories(headers/)

file (GLOB_RECURSE SRC_FILES src/*.cc)

add_executable(Optimizer ${SRC_FILES})