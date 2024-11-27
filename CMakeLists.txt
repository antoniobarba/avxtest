cmake_minimum_required(VERSION 3.30)
project(AVXTest VERSION 1.0)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED 23)

add_compile_options(-ffast-math)

find_package(SDL2 REQUIRED)

add_subdirectory(mandelbrot)

add_executable(mand src/main.cc )
add_executable(test src/testlib.cc)

if(USE_GPU)
    add_compile_options(-DUSE_GPU)
    target_compile_definitions(mand PRIVATE CL_TARGET_OPENCL_VERSION=100)
    target_compile_definitions(test PRIVATE CL_TARGET_OPENCL_VERSION=100)
endif(USE_GPU)

target_include_directories(mand PUBLIC include SDL2_INCLUDE_DIRS)
target_link_libraries(mand PUBLIC SDL2::SDL2 mandelbrot)
target_link_libraries(test mandelbrot)