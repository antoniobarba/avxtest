#pragma once
#include <easycl.h>

void mandelbrot_opencl(easycl::Device& device, void *points, int w, int h);