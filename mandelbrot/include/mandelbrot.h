#pragma once

#ifdef __cplusplus
extern "C" {
#endif

    // points must be allocated with at least 4 x w x h bytes 
    extern void mandelbrot_cpu(void * points, int w, int h);

    // points must be allocated with at least 4 x w x h bytes
    extern void mandelbrot_gpu(void * kernel, void * points, int w, int h);
    extern void * mandelbrot_create_gpu_kernel();
    extern void mandelbrot_free_gpu_kernel(void *kernel);

#ifdef __cplusplus
}
#endif
