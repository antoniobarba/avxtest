#pragma once
#include <easycl.h>

void mandelbrot_opencl(easycl::Kernel& kernel, easycl::Device& queue, void *points, int w, int h);