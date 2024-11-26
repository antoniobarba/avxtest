#pragma once
#include <hwy/base.h>
#include <hwy/detect_targets.h>

namespace mandelbrot_highway {

    void mandelbrot(void* HWY_RESTRICT points, int w, int h);

}