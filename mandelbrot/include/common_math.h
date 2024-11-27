#pragma once

#include <chrono>
#include <functional>
#include <omp.h>
#include <openacc.h>
#include <iostream>


template <typename T>
T map_to(T value, T sourceMin, T sourceMax, T destMin, T destMax)
{
    T s = sourceMax - sourceMin;
    T d = destMax - destMin;
    T sourceRatio = (value - sourceMin) / s;
    return destMin + sourceRatio * d;
}

inline uint32_t map_rgba(uint8_t r, uint8_t g, uint8_t b, uint8_t a)
{
    return r << 24 | g << 16 | b << 8 | a;
}

inline uint32_t map_argb(uint8_t r, uint8_t g, uint8_t b, uint8_t a)
{
    return a << 24 | r << 16 | g << 8 | b;
}

using MandelbrotFunc = std::function<void(void*, int, int)>;

template <class T>
void mandelbrot_base(void* points, int w, int h)
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
            p[j * w + i] = map_argb((uint8_t)color, (uint8_t)color, (uint8_t)color, 255);
        }
    }
}

template <class T>
void mandelbrot_omp(void* points, int w, int h)
{
    uint32_t * p = (uint32_t *)points;      
    #pragma omp parallel for default(private) shared(w, h, p)
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
            p[j * w + i] = map_argb((uint8_t)color, (uint8_t)color, (uint8_t)color, 255);
        }
    }
}


template <class T>
void mandelbrot_omp_gpu(void* points, int w, int h)
{
    uint32_t * p = (uint32_t *)points;      
    #pragma omp target teams distribute parallel for default(private) shared(w, h, p)
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
            p[j * w + i] = map_argb((uint8_t)color, (uint8_t)color, (uint8_t)color, 255);
        }
    }
}


template <class T>
void mandelbrot_acc_gpu(void* points, int w, int h)
{
    uint32_t * p = (uint32_t *)points;      
    #pragma acc parallel loop default(private) shared(w, h, p)
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
            p[j * w + i] = map_argb((uint8_t)color, (uint8_t)color, (uint8_t)color, 255);
        }
    }
}