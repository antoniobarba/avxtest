#pragma once
#include <hwy/base.h>

namespace mandelbrot_highway {

    void mandelbrot(void* HWY_RESTRICT points, int w, int h);

    void mandelbrot_sse2(void* HWY_RESTRICT points, int w, int h);

    void mandelbrot_ssse3(void* HWY_RESTRICT points, int w, int h);

    void mandelbrot_sse4(void* HWY_RESTRICT points, int w, int h);

    void mandelbrot_avx2(void* HWY_RESTRICT points, int w, int h);

    void mandelbrot_avx512(void* HWY_RESTRICT points, int w, int h);
}