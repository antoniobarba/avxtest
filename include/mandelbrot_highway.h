#pragma once
#include <hwy/base.h>

namespace mandelbrot_highway {

    void mandelbrot(void* HWY_RESTRICT points, int w, int h);
}