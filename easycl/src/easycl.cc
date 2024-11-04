#include <easycl.h>
#include <iostream>

using namespace easycl;

Device::Device(cl_platform_id platformId, cl_device_id deviceId)
{
    _platform = platformId;
    _device = deviceId;

    cl_int err;
    _context = clCreateContext(NULL, 1, &_device, NULL, NULL, &err);

    if (err != 0)
    {
        _status = DeviceStatus::ERROR_CREATING_CONTEXT;
        return;
    }

    _queue = clCreateCommandQueueWithProperties(_context, _device, nullptr, &err);

    if (err != 0)
    {
        _status = DeviceStatus::ERROR_CREATING_COMMAND_QUEUE;
        return;
    }

    _status = DeviceStatus::READY;
}

Device::~Device()
{
    clReleaseKernel(_kernel);
    clReleaseCommandQueue(_queue);
    clReleaseProgram(_program);
    clReleaseContext(_context);
}

bool Device::loadProgram(const std::filesystem::path &program, const std::string &kernel)
{
    FILE *program_handle = fopen(program.c_str(), "r");

    fseek(program_handle, 0, SEEK_END);
    size_t program_size = ftell(program_handle);
    rewind(program_handle);
    char *program_buffer = (char *)malloc(program_size + 1);
    program_buffer[program_size] = '\0';
    fread(program_buffer, sizeof(char), program_size, program_handle);
    fclose(program_handle);
    cl_int err;
    _program = clCreateProgramWithSource(_context, 1, (const char **)&program_buffer, &program_size, &err);
    free(program_buffer);

    if (err != 0)
    {
        _status = DeviceStatus::ERROR_CREATING_PROGRAM;
        return false;
    }

    err = clBuildProgram(_program, 0, NULL, NULL, NULL, NULL);
    if (err != 0)
    {
        std::cerr << "OpenCL Build failed\n";
        size_t length;
        char buffer[128 * 1024];
        clGetProgramBuildInfo(_program, _device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &length);
        std::cerr << "--- Build log ---\n " << buffer << std::endl;
        _status = DeviceStatus::ERROR_BUILDING_PROGRAM;
        return false;
    }

    _kernel = clCreateKernel(_program, kernel.c_str(), &err);
    if (err != 0)
    {
        _status = DeviceStatus::ERROR_CREATING_KERNEL;
        return false;
    }

    _status = DeviceStatus::PROGRAM_LOADED;
    return true;
}

void Device::runKernel(const std::vector<size_t> &dimensions)
{
    clEnqueueNDRangeKernel(_queue, _kernel, dimensions.size(), NULL, dimensions.data(), NULL, 0, NULL, NULL);
}

EasyCL::EasyCL()
{
}

Device EasyCL::getFirstDevice()
{
    cl_platform_id platform;
    cl_device_id device;
    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    return Device(platform, device);
}