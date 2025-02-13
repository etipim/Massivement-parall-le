cmake_minimum_required(VERSION 3.26)
project(Bat_Algorithm)

set(CMAKE_CXX_STANDARD 17)

add_executable(Bat_Algorithm main.cpp
        WOA.cpp
        WOA.h
        BA.cpp
        BA.h
        Agent.cpp
        Agent.h
        kernel.cpp
        kernel.cu
        kernel.h)
