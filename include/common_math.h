#pragma once

#include <chrono>
#include <functional>
#include <omp.h>
#include <iostream>


template <typename T>
T map_to(T value, T sourceMin, T sourceMax, T destMin, T destMax)
{
    T s = sourceMax - sourceMin;
    T d = destMax - destMin;
    T sourceRatio = (value - sourceMin) / s;
    return destMin + sourceRatio * d;
}

using MandelbrotFunc = std::function<void(void*, int, int, const SDL_PixelFormat*)>;

/* Mandelbrot set pseudo-code from Wikipedia: 

    for each pixel (Px, Py) on the screen do
    x0 := scaled x coordinate of pixel (scaled to lie in the Mandelbrot X scale (-2.00, 0.47))
    y0 := scaled y coordinate of pixel (scaled to lie in the Mandelbrot Y scale (-1.12, 1.12))
    x := 0.0
    y := 0.0
    iteration := 0
    max_iteration := 1000
    while (x*x + y*y ≤ 2*2 AND iteration < max_iteration) do
        xtemp := x*x - y*y + x0
        y := 2*x*y + y0
        x := xtemp
        iteration := iteration + 1
 
    color := palette[iteration]
    plot(Px, Py, color)
*/

template <class T>
void mandelbrot_base(void* points, int w, int h, const SDL_PixelFormat* f)
{
    uint32_t * p = (uint32_t *)points; 

    for (int j=0; j<h; ++j)
    {
        for (int i=0; i<w; ++i)
        {
            T x0 = map_to<T>(i, 0, w, -2.0, 0.47);
            T y0 = map_to<T>(j, 0, h, -1.12, 1.12);
            T x = 0;
            T y = 0;
            int iteration = 0;
            const int max_iteration = 1000;
            while (x*x + y*y <= 4 && iteration < max_iteration)
            {
                T xtemp = x*x - y*y + x0;
                y = 2*x*y + y0;
                x = xtemp;
                ++iteration;
            }
            T color = map_to(iteration, 0, 15, 0, 255);
            p[j * w + i] = SDL_MapRGBA(f, (Uint8)color, (Uint8)color, (Uint8)color, 255);
        }
    }
}

template <class T>
void mandelbrot_omp(void* points, int w, int h, const SDL_PixelFormat* f)
{
    uint32_t * p = (uint32_t *)points;      
    #pragma omp parallel for default(private) shared(w, h, f, p)
    for (int j=0; j<h; ++j)
    {
        for (int i=0; i<w; ++i)
        {
            T x0 = map_to<T>(i, 0, w, -2.0, 0.47);
            T y0 = map_to<T>(j, 0, h, -1.12, 1.12);
            T x = 0;
            T y = 0;
            int iteration = 0;
            const int max_iteration = 1000;
            while (x*x + y*y <= 4 && iteration < max_iteration)
            {
                T xtemp = x*x - y*y + x0;
                y = 2*x*y + y0;
                x = xtemp;
                ++iteration;
            }
            T color = map_to(iteration, 0, 15, 0, 255);
            p[j * w + i] = SDL_MapRGBA(f, (Uint8)color, (Uint8)color, (Uint8)color, 255);
        }
    }
}
