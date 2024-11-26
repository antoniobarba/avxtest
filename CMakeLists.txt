cmake_minimum_required(VERSION 3.30)
project(AVXTest VERSION 1.0)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED 23)
option(USE_GPU "Enable GPU computing using OpenCL" Off)

add_compile_options(-ffast-math)
include(CheckSourceCompiles)

# Check if compiling on PowerPC
check_source_compiles(CXX
"#if defined(__powerpc64__) || defined(_M_PPC) || defined(__powerpc__)
#define MY_ARCH_PPC 1
#else
#define MY_ARCH_PPC 0
#endif
#if !MY_ARCH_PPC
#error \"No PPC arch\"
#endif
int main() { return 0; }
"

HAVE_PPC)
# End PPC

if(HAVE_PPC)
    add_compile_options(-maltivec)
endif(HAVE_PPC)

find_package(OpenMP REQUIRED)
find_package(SDL2 REQUIRED)
find_package(HWY REQUIRED NO_MODULE)

if(USE_GPU)
    add_compile_options(-DUSE_GPU)
    add_subdirectory(easycl)
    set(EasyCL_LIBRARY EasyCL)
else(USE_GPU)
    set(OpenCL_INCLUDE_DIR "")
    set(OpenCL_LIBRARY "")
    set(EasyCL_LIBRARY "")
endif(USE_GPU)

add_executable(test src/mandelbrot_highway.cc src/mandelbrot_opencl.cc src/main.cc )
target_include_directories(test PUBLIC include SDL2_INCLUDE_DIRS ${HWY_INCLUDE_DIRS} ${OpenCL_INCLUDE_DIR})
target_link_libraries(test PUBLIC OpenMP::OpenMP_CXX ${OpenCL_LIBRARY} SDL2::SDL2 hwy::hwy ${EasyCL_LIBRARY})
target_compile_definitions(test PRIVATE CL_TARGET_OPENCL_VERSION=100)