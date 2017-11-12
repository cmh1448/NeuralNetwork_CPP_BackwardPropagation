cmake_minimum_required(VERSION 3.8)
project(NeuralNetwork)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES
        Libraries/Matrix.cpp
        Libraries/Matrix.hpp
        Libraries/Network.cpp
        Libraries/Network.h
        main.cpp)

add_executable(NeuralNetwork ${SOURCE_FILES})