#include <mandelbrot.h>
#include <common_math.h>
#include <mandelbrot_highway.h>

#if USE_GPU
#include <mandelbrot_opencl.h>
#include <easycl.h>
#include <CL/cl.h>
#endif


// points must be allocated with at least 4 x w x h bytes 
void mandelbrot_cpu(void * points, int w, int h)
{
    mandelbrot_highway::mandelbrot(points, w, h);
}

struct __dev_ker
{
    easycl::Device* dev;
    easycl::Kernel* ker;
};

void* mandelbrot_create_gpu_kernel()
{
#if USE_GPU
    auto clDevices = easycl::getDevices(CL_DEVICE_TYPE_GPU);
    for (auto d : clDevices)
    {
        if (d.getVendor() != "Intel(R) Corporation")
        {
            std::string programSource = 
            "float map_to(float value, float sourceMin, float sourceMax, float destMin, float destMax)\n\
            {\n\
                float s = sourceMax - sourceMin;\n\
                float d = destMax - destMin;\n\
                float sourceRatio = (value - sourceMin) / s;\n\
                return destMin + sourceRatio * d;\n\
            }\n\
            \n\
            uint map_argb(uchar r, uchar g, uchar b, uchar a)\n\
            {\n\
                // return 0xffffffff;\n\
                return a << 24 | r << 16 | g << 8 | b;\n\
            }\n\
            \n\
            __kernel void calc_pixel(__global void *buffer, int w, int h)\n\
            {\n\
                int i = get_global_id(0);\n\
                int j = get_global_id(1);\n\
                float x0 = map_to(i, 0, w, -2.0, 0.47);\n\
                float y0 = map_to(j, 0, h, -1.12, 1.12);\n\
                float x = 0;\n\
                float y = 0;\n\
                int iteration = 0;\n\
                const int max_iteration = 1000;\n\
                while (x * x + y * y <= 4 && iteration < max_iteration)\n\
                {\n\
                    float xtemp = x * x - y * y + x0;\n\
                    y = 2 * x * y + y0;\n\
                    x = xtemp;\n\
                    ++iteration;\n\
                }\n\
                float color = map_to(iteration, 0, 15, 0, 255);\n\
                __global uint * p = (__global uint *)buffer; \n\
                p[j * w + i] = map_argb((uchar)color, (uchar)color, (uchar)color, 255);\n\
            }\n";
            __dev_ker *ret = new __dev_ker{ new easycl::Device(d), new easycl::Kernel(d.loadProgram(programSource).createKernel("calc_pixel"))};
            
            return (void*)ret;
        }
    }
    return NULL;
#else
    return NULL;
#endif
}

void mandelbrot_free_gpu_kernel(void *kernel)
{
    __dev_ker * p = (__dev_ker*)kernel;
    delete (p->ker);
    delete (p->dev);
    delete p;
}

// points must be allocated with at least 4 x w x h bytes
void mandelbrot_gpu(void* kernel, void * points, int w, int h)
{
#if USE_GPU
    __dev_ker * p = (__dev_ker*)kernel;
    mandelbrot_opencl(*p->ker, *p->dev, points, w, h);        
#endif
}