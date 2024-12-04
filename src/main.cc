#include <iostream>
#include <cstring>
#include <SDL2/SDL.h>
#include <mandelbrot_highway.h>
#if USE_GPU
#include <mandelbrot_opencl.h>
#include <CL/cl.h>
#include <easycl.h>
#endif

#include <common_math.h>
#include <thread>
#include <hwy/targets.h>

constexpr size_t window_w = 4096;
constexpr size_t window_h = 4096;

using namespace mandelbrot_highway;

void time_and_test(size_t iterations, MandelbrotFunc f, const std::string &name, void *buffer, int w, int h)
{
    std::cout << "Testing " << name << " over " << iterations << " iterations: " << std::flush;

    using ns = std::chrono::nanoseconds;
    ns bestTime{std::chrono::nanoseconds::max()};
    ns worst{std::chrono::nanoseconds::min()};

    for (int i = 0; i < iterations; ++i)
    {
        auto time_before = std::chrono::high_resolution_clock::now();
        f(buffer, w, h);
        auto elapsed = std::chrono::high_resolution_clock::now() - time_before;

        if (elapsed < bestTime)
        {
            bestTime = elapsed;
        }
        if (elapsed > worst)
        {
            worst = elapsed;
        }
    }

    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(bestTime) << " - " << std::chrono::duration_cast<std::chrono::milliseconds>(worst) << "\n";
}

int main(int argc, char **argv)
{
    bool render = true;
    if (argc == 2 && strcmp(argv[1], "-norender") == 0)
    {
        render = false;
    }
        
    void *buffer;
    SDL_PixelFormat *format = nullptr;
    SDL_Surface *canvas = nullptr;
    SDL_Surface *s = nullptr;
    SDL_Window *window = nullptr;

    if (render)
    {
        // Init video
        SDL_Init(SDL_INIT_VIDEO);

        window = SDL_CreateWindow("SDL2Test", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, window_w, window_h, 0);
        if (!window)
            return 0;

        // Calculate shit
        s = SDL_GetWindowSurface(window);
        canvas = SDL_CreateRGBSurfaceWithFormat(
            0, window_w, window_h, 32, SDL_PIXELFORMAT_ARGB8888);

        if (SDL_MUSTLOCK(canvas))
        {
            SDL_LockSurface(canvas);
        }
        buffer = canvas->pixels;
        format = canvas->format;
    }
    else
    {
        buffer = malloc(sizeof(size_t) * window_h * window_w);
    }

    bool quit = false;

    const int howMany = 5;
    const int ncpus = omp_get_num_procs();
    time_and_test(howMany, mandelbrot_base<float>,"Base algo single thread", buffer, window_w, window_h);
    time_and_test(howMany, mandelbrot_omp<float>, "Base algo on " + std::to_string(ncpus) + " threads", buffer, window_w, window_h);

    #if USE_GPU
    auto clDevices = easycl::getDevices(CL_DEVICE_TYPE_GPU);
    for (auto d : clDevices)
    {
        if (d.getVendor() != "Intel(R) Corporation")
        {
            //easycl::Kernel k = d.loadProgramFromFile("../src/mandelbrot_opencl_program.cl").createKernel("calc_pixel");
            
            std::string programSource = 
            "float map_to(float value, float sourceMin, float sourceMax, float destMin, float destMax)\n\
            {\n\
                float s = sourceMax - sourceMin;\n\
                float d = destMax - destMin;\n\
                float sourceRatio = (value - sourceMin) / s;\n\
                return destMin + sourceRatio * d;\n\
            }\n\
            \n\
            uint map_rgba(uchar r, uchar g, uchar b, uchar a)\n\
            {\n\
                // return 0xffffffff;\n\
                return r << 24 | g << 16 | b << 8 | a;\n\
            }\n\
            \n\
            __kernel void calc_pixel(__global void *buffer, int w, int h)\n\
            {\n\
                int i = get_global_id(0);\n\
                int j = get_global_id(1);\n\
                float x0 = map_to(i, 0, w, -2.0, 0.47);\n\
                float y0 = map_to(j, 0, h, -1.12, 1.12);\n\
                float x = 0;\n\
                float y = 0;\n\
                int iteration = 0;\n\
                const int max_iteration = 1000;\n\
                while (x * x + y * y <= 4 && iteration < max_iteration)\n\
                {\n\
                    float xtemp = x * x - y * y + x0;\n\
                    y = 2 * x * y + y0;\n\
                    x = xtemp;\n\
                    ++iteration;\n\
                }\n\
                float color = map_to(iteration, 0, 15, 0, 255);\n\
                __global uint * p = (__global uint *)buffer; \n\
                p[j * w + i] = map_rgba((uchar)color, (uchar)color, (uchar)color, 255);\n\
            }\n";
            
            easycl::Kernel k = d.loadProgram(programSource).createKernel("calc_pixel");
            auto mandelbrotToCall = [&](void *points, int w, int h){
                mandelbrot_opencl(k, d, points, w, h);
            };
            time_and_test(howMany, mandelbrotToCall, d.getName(), buffer, window_w, window_h);
        }
    }
    #endif

    auto allTargets = hwy::SupportedAndGeneratedTargets();
    for (int64_t target : allTargets)
    {
        // Enable one target at a time
        hwy::SetSupportedTargetsForTest(target);
        time_and_test(howMany, mandelbrot_highway::mandelbrot, std::string(hwy::TargetName(target)) + " on " + std::to_string(ncpus) + " threads", buffer, window_w, window_h);
    }

    // Reset to normal
    hwy::SetSupportedTargetsForTest(0);

    if (render)
    {
        if (SDL_MUSTLOCK(canvas))
        {
            SDL_UnlockSurface(canvas);
        }

        // Cleanup
        SDL_Event event;
        while (!quit)
        {
            while (SDL_PollEvent(&event))
            {
                switch (event.type)
                {
                case SDL_QUIT:
                {
                    quit = true;
                    break;
                }
                case SDL_KEYDOWN:
                {
                    if (event.key.keysym.sym == SDLK_q ||
                        event.key.keysym.sym == SDLK_ESCAPE)
                    {
                        quit = true;
                    }
                    break;
                }
                }
            }

            if (SDL_MUSTLOCK(canvas))
            {
                SDL_LockSurface(canvas);
            }

            SDL_BlitSurface(canvas, 0, s, 0);
            SDL_UpdateWindowSurface(window);

            if (SDL_MUSTLOCK(canvas))
            {
                SDL_UnlockSurface(canvas);
            }

            SDL_Delay(10);
        }

        SDL_Quit();
    }
    else
    {
        free(buffer);
    }

    std::cout << std::endl;
}
