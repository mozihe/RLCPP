cmake_minimum_required(VERSION 3.30)

set(CMAKE_CXX_STANDARD 20)

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

add_library(net base_net.cpp base_net.hpp)

target_link_libraries(net Eigen3::Eigen)

target_include_directories(net PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})