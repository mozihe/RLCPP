cmake_minimum_required(VERSION 3.30)
project(RLCPP_Env)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(OpenCV REQUIRED)
find_library(ZMQ_LIBRARY NAMES zmq)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

file(GLOB SOURCES_DIR "src/*.cpp")

add_executable(RLCPP_Env main.cpp ${SOURCES_DIR})

target_link_libraries(RLCPP_Env ${OpenCV_LIBS} ${ZMQ_LIBRARY})
