#pragma once

#include <filesystem>
#include <vector>
#include <CL/cl.h>

namespace easycl {
    class Program;
    
    enum class DeviceStatus
    {
        UNINITIALIZED,
        READY,
        ERROR_CREATING_CONTEXT,
        ERROR_CREATING_COMMAND_QUEUE
    };

    class Device
    {
    public:
        Device(cl_platform_id platformId, cl_device_id deviceId);
        ~Device();

        Program loadProgramFromFile(const std::filesystem::path &program);
        Program loadProgram(const std::string &program);

        std::string getName() const { return _name; }
        std::string getVendor() const { return _vendor; }

        cl_command_queue getQueue() const { return _queue; }
        cl_context getContext() const { return _context; }
        cl_device_id getDevice() const { return _device; }
        DeviceStatus getStatus() const { return _status; }

    private:
        

        cl_platform_id _platform;
        cl_device_id _device;
        cl_context _context=nullptr;
        cl_command_queue _queue=nullptr;
        cl_program _program=nullptr;
        cl_kernel _kernel=nullptr;

        std::string _name;
        std::string _vendor;

        DeviceStatus _status;
    };
}
    