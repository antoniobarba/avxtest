#pragma once
#include <hwy/base.h>
class SDL_PixelFormat;

namespace mandelbrot_highway {

    void mandelbrot(void* HWY_RESTRICT points, int w, int h, const SDL_PixelFormat* format);
}