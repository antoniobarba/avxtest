#include <common_math.h>
#include <easycl.h>
#include <CL/cl.h>

void mandelbrot_opencl(void *points, int w, int h)
{
    static bool init = false;
    static easycl::Device d = easycl::EasyCL::getFirstDevice();

    if (!init)
    {
        d.loadProgram("../src/mandelbrot_opencl_program.cl", "calc_pixel");
    }

    size_t bufSize = sizeof(uint32_t) * w * h;
    cl_mem clBuffer = clCreateBuffer(d.getContext(), CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, bufSize, points, nullptr);

    d.runKernel(std::vector<size_t>{(uint32_t)w, (uint32_t)h}, clBuffer, w, h);

    clEnqueueReadBuffer(d.getQueue(), clBuffer, CL_TRUE, 0, bufSize, points, 0, NULL, NULL);

    clReleaseMemObject(clBuffer);
}