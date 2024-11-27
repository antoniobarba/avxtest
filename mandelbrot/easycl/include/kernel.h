#pragma once
#include <CL/cl.h>
#include <vector>

namespace easycl
{

    enum class KernelStatus
    {
        UNDEFINED,
        READY
    };

    class Kernel
    {
    public:
        Kernel() { _status = KernelStatus::READY; }
        Kernel(cl_kernel kernel);

        void run(cl_command_queue queue, const std::vector<size_t> &dimensions);

        template <typename... Args>
        void run(cl_command_queue queue, const std::vector<size_t> &dimensions, Args &&...args)
        {
            setKernelArgs(0, std::forward<Args>(args)...);
            run(queue, dimensions);
        }

    private:
        template <typename T>
        void setKernelArgs(size_t numOfArgs, T &&value)
        {
            clSetKernelArg(_kernel, numOfArgs, sizeof(T), &value);
        }

        template <typename T, typename... Args>
        void setKernelArgs(size_t numOfArgs, T &&value, Args &&...args)
        {
            clSetKernelArg(_kernel, numOfArgs, sizeof(T), &value);
            setKernelArgs(numOfArgs + 1, std::forward<Args>(args)...);
        }

        cl_kernel _kernel;
        KernelStatus _status;
    };
}