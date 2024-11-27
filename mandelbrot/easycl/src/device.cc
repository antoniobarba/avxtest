#include <device.h>
#include <iostream>
#include <program.h>
using namespace easycl;

Device::Device(cl_platform_id platformId, cl_device_id deviceId)
{
    _platform = platformId;
    _device = deviceId;

    size_t valueSize;
    clGetDeviceInfo(_device, CL_DEVICE_NAME, 0, NULL, &valueSize);
    char *value = (char *)malloc(valueSize + 1);
    value[valueSize] = 0;
    clGetDeviceInfo(_device, CL_DEVICE_NAME, valueSize, value, NULL);
    _name.assign(value);
    free(value);

    clGetDeviceInfo(_device, CL_DEVICE_VENDOR, 0, NULL, &valueSize);
    value = (char *)malloc(valueSize + 1);
    value[valueSize] = 0;
    clGetDeviceInfo(_device, CL_DEVICE_VENDOR, valueSize, value, NULL);
    _vendor.assign(value);
    free(value);

    cl_int err;
    _context = clCreateContext(NULL, 1, &_device, NULL, NULL, &err);

    if (err != 0)
    {
        _status = DeviceStatus::ERROR_CREATING_CONTEXT;
        return;
    }

    // _queue = clCreateCommandQueueWithProperties(_context, _device, nullptr, &err);
    _queue = clCreateCommandQueue(_context, _device, 0, &err);

    if (err != 0)
    {
        _status = DeviceStatus::ERROR_CREATING_COMMAND_QUEUE;
        return;
    }

    _status = DeviceStatus::READY;
}

Device::~Device()
{
    if (_kernel)
        clReleaseKernel(_kernel);

    if (_program)
        clReleaseProgram(_program);

    // if (_queue)
    //     clReleaseCommandQueue(_queue);
    
    if (_context)
        clReleaseContext(_context);
}

Program Device::loadProgramFromFile(const std::filesystem::path &program)
{
    auto p = Program(*this, program);
    return p;
}

Program Device::loadProgram(const std::string &program)
{
    auto p = Program(*this, program);
    return p;
}