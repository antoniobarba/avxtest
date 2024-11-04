#include <program.h>
#include <iostream>
#include <device.h>
#include <kernel.h>

using namespace easycl;

Program::Program(Device &device, const std::filesystem::path &program)
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
    _program = clCreateProgramWithSource(device.getContext(), 1, (const char **)&program_buffer, &program_size, &err);
    free(program_buffer);

    if (err != 0)
    {
        _status = ProgramStatus::FILE_ERROR;
        return;
    }

    err = clBuildProgram(_program, 0, NULL, NULL, NULL, NULL);
    if (err != 0)
    {
        std::cerr << "OpenCL Build failed\n";
        size_t length;
        char buffer[128 * 1024];
        clGetProgramBuildInfo(_program, device.getDevice(), CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &length);
        std::cerr << "--- Build log ---\n " << buffer << std::endl;
        _status = ProgramStatus::BUILD_ERROR;
        return;
    }

    _status = ProgramStatus::READY;
}

Kernel Program::createKernel(const std::string &name)
{
    cl_int err;
    cl_kernel kernel = clCreateKernel(_program, name.c_str(), &err);
    if (err != 0)
    {
        return Kernel();
    }
    
    return Kernel(kernel);
}