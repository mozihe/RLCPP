cmake_minimum_required(VERSION 3.30)
project(RLAgent)

set(CMAKE_CXX_STANDARD 20)

find_package(Eigen3 REQUIRED)
find_library(ZMQ_LIBRARY NAMES zmq)

include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_subdirectory(net)

file(GLOB SOURCES_DIR "src/*.cpp")

add_executable(RLAgent main.cpp ${SOURCES_DIR})

target_link_libraries(RLAgent Eigen3::Eigen net ${ZMQ_LIBRARY})
