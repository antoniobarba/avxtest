#include <kernel.h>

using namespace easycl;

Kernel::Kernel(cl_kernel kernel)
{
    _kernel = kernel;
    _status = KernelStatus::READY;
}

void Kernel::run(cl_command_queue queue, const std::vector<size_t> &dimensions)
{
    clEnqueueNDRangeKernel(queue, _kernel, dimensions.size(), NULL, dimensions.data(), NULL, 0, NULL, NULL);
}