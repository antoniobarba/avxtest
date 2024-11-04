#pragma once

#include <device.h>
#include <program.h>
#include <kernel.h>
#include <vector>

namespace easycl
{
    Device getFirstDevice(cl_device_type deviceType);
    std::vector<Device> getDevices(cl_device_type deviceType);
}