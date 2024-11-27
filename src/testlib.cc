#include  <mandelbrot.h>

int main()
{
    // Create buffer
    const int w = 128;
    const int h = 128;
    char buffer[4*w*h];
    void * kernel = mandelbrot_create_gpu_kernel();
    mandelbrot_gpu(kernel, &buffer, w, h);
    mandelbrot_free_gpu_kernel(kernel);
    return 0;
}