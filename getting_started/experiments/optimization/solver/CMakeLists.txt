cmake_minimum_required(VERSION 3.22)
project(solver)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

include_directories(headers/)

file (GLOB_RECURSE SRC_FILES src/*.cc)
add_executable(solver ${SRC_FILES})

find_package(MPI REQUIRED)
target_link_libraries(solver ${MPI_LIBRARIES})