#pragma once

#include <filesystem>
#include <vector>
#include <CL/cl.h>

namespace easycl {
    enum class DeviceStatus
    {
        UNINITIALIZED,
        READY,
        ERROR_CREATING_CONTEXT,
        ERROR_CREATING_COMMAND_QUEUE,
        ERROR_CREATING_PROGRAM,
        ERROR_BUILDING_PROGRAM,
        ERROR_CREATING_KERNEL,
        PROGRAM_LOADED
    };

    class Device
    {
    public:
        Device(cl_platform_id platformId, cl_device_id deviceId);
        ~Device();

        bool loadProgram(const std::filesystem::path &program, const std::string &kernel);

        void runKernel(const std::vector<size_t> &dimensions);

        template <typename... Args>
        void runKernel(const std::vector<size_t> &dimensions, Args &&...args)
        {
            setKernelArgs(0, std::forward<Args>(args)...);
            runKernel(dimensions);
        }

        std::string getName() const;

        cl_command_queue getQueue() const { return _queue; }
        cl_context getContext() const { return _context; }

    private:
        template <typename T>
        void setKernelArgs(size_t numOfArgs, T&& value)
        {
            clSetKernelArg(_kernel, numOfArgs, sizeof(T), &value);
        }

        template <typename T, typename... Args>
        void setKernelArgs(size_t numOfArgs, T&& value, Args&& ... args)
        {
            clSetKernelArg(_kernel, numOfArgs, sizeof(T), &value);
            setKernelArgs(numOfArgs+1, std::forward<Args>(args)...);
        }

        cl_platform_id _platform;
        cl_device_id _device;
        cl_context _context=nullptr;
        cl_command_queue _queue=nullptr;
        cl_program _program=nullptr;
        cl_kernel _kernel=nullptr;

        std::string _name;

        DeviceStatus _status;
    };
}
    