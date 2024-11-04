#include <common_math.h>
#include <mandelbrot_opencl.h>
#include <easycl.h>
#include <CL/cl.h>

void mandelbrot_opencl(easycl::Device& device, void *points, int w, int h)
{
    size_t bufSize = sizeof(uint32_t) * w * h;
    cl_mem clBuffer = clCreateBuffer(device.getContext(), CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, bufSize, points, nullptr);

    device.runKernel(std::vector<size_t>{(uint32_t)w, (uint32_t)h}, clBuffer, w, h);

    clEnqueueReadBuffer(device.getQueue(), clBuffer, CL_TRUE, 0, bufSize, points, 0, NULL, NULL);

    clReleaseMemObject(clBuffer);
}