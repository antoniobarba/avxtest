cmake_minimum_required(VERSION 3.30)
project(EasyCL VERSION 1.0)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED 23)

find_package(OpenCL REQUIRED)

add_library(EasyCL SHARED src/easycl.cc src/device.cc src/program.cc src/kernel.cc)

target_include_directories(EasyCL PUBLIC include PRIVATE ${OpenCL_INCLUDE_DIR})
target_compile_definitions(EasyCL PRIVATE CL_TARGET_OPENCL_VERSION=100)