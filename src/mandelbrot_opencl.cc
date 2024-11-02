#include <common_math.h>
#include <CL/cl.h>

void mandelbrot_opencl(void *points, int w, int h)
{
    static bool init = false;
    static cl_platform_id platform;
    static cl_device_id device;
    static cl_context context;
    static cl_command_queue queue;
    static cl_program program;
    static cl_kernel kernel;


    char *program_buffer, *program_log;
    size_t program_size, log_size;
    cl_int i, err;

    if (!init)
    {
        clGetPlatformIDs(1, &platform, NULL);

        clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
        
        context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
        FILE *program_handle = fopen("../src/mandelbrot_opencl_program.cl", "r");
        fseek(program_handle, 0, SEEK_END);
        program_size = ftell(program_handle);
        rewind(program_handle);
        program_buffer = (char *)malloc(program_size + 1);
        program_buffer[program_size] = '\0';
        fread(program_buffer, sizeof(char), program_size, program_handle);
        fclose(program_handle);
        program = clCreateProgramWithSource(context, 1, (const char **)&program_buffer, &program_size, &err);
        free(program_buffer);
        err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
        if (err != 0)
        {
            std::cout << "OpenCL Build failed\n";
            size_t length;
            char buffer[128 * 1024];
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &length);
            std::cout << "--- Build log ---\n " << buffer << std::endl;
            exit(1);
        }

        kernel = clCreateKernel(program, "calc_pixel", &err);
        queue = clCreateCommandQueue(context, device, 0, &err);

        init = true;
    }
    size_t bufSize = sizeof(uint32_t) * w * h;
    cl_mem clBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, bufSize, points, &err);
    cl_int clW = w;
    cl_int clH = h;

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &clBuffer);
    clSetKernelArg(kernel, 1, sizeof(cl_int), &clW);
    clSetKernelArg(kernel, 2, sizeof(cl_int), &clH);

    const size_t work_units_per_kernel[2] = {(size_t)w, (size_t)h};

    clEnqueueNDRangeKernel(queue, kernel, 2, NULL, work_units_per_kernel, NULL, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, clBuffer, CL_TRUE, 0, bufSize, points, 0, NULL, NULL);

    clReleaseMemObject(clBuffer);
    // clReleaseKernel(kernel);
    // clReleaseCommandQueue(queue);
    // clReleaseProgram(program);
    // clReleaseContext(context);
}