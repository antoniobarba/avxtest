cmake_minimum_required(VERSION 3.30)
project(AVXTest VERSION 1.0)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED 23)

set(CMAKE_CXX_FLAGS "-ffast-math")

find_package(OpenACC REQUIRED)
find_package(OpenMP REQUIRED)
find_package(SDL2 REQUIRED)
find_package(HWY REQUIRED NO_MODULE)
find_package(OpenCL REQUIRED)

add_executable(test src/mandelbrot_highway.cc src/mandelbrot_opencl.cc src/main.cc )
target_include_directories(test PUBLIC include SDL2_INCLUDE_DIRS ${HWY_INCLUDE_DIRS} ${OpenCL_INCLUDE_DIR})
target_link_libraries(test PUBLIC OpenMP::OpenMP_CXX ${OpenCL_LIBRARY} SDL2::SDL2 hwy::hwy)
target_compile_definitions(test PRIVATE CL_TARGET_OPENCL_VERSION=100)