#include <easycl.h>

using namespace easycl;

Device easycl::getFirstDevice(cl_device_type deviceType)
{
    cl_platform_id platform;
    cl_device_id device;
    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, deviceType, 1, &device, NULL);

    return Device(platform, device);
}

std::vector<Device> easycl::getDevices(cl_device_type deviceType)
{
    std::vector<Device> ret;

    // Get number of platforms
    cl_uint numPlatforms;
    clGetPlatformIDs(0, nullptr, &numPlatforms);

    ret.reserve(numPlatforms);

    // Get all the platforms
    cl_platform_id *platforms = new cl_platform_id[numPlatforms];
    clGetPlatformIDs(numPlatforms, platforms, nullptr);

    // List all devices under each platform
    for (cl_uint p = 0; p < numPlatforms; ++p)
    {
        // Get device number
        cl_uint numDevices;
        clGetDeviceIDs(platforms[p], deviceType, 0, nullptr, &numDevices);

        // Get all devices
        cl_device_id *devices = new cl_device_id[numDevices];
        clGetDeviceIDs(platforms[p], deviceType, numDevices, devices, nullptr);

        for (cl_uint d = 0; d < numDevices; ++d)
        {
            ret.emplace_back(platforms[p], devices[d]);
        }
        delete[] devices;
    }
    delete[] platforms;

    return ret;
}