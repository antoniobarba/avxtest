#pragma once

#include <CL/cl.h>
#include <filesystem>

namespace easycl
{
    class Device;
    class Kernel;

    enum class ProgramStatus
    {
        UNDEFINED,
        READY,
        BUILD_ERROR,
        FILE_ERROR,
        KERNEL_ERROR
    };

    class Program
    {
    public:
        Program(Device &device, const std::filesystem::path &program);
        Program(Device &device, const std::string &programSource);

        ProgramStatus getStatus() const { return _status; }

        Kernel createKernel(const std::string &name);

    private:
        void build(Device &device);

        cl_program _program;
        ProgramStatus _status;
    };

    
}