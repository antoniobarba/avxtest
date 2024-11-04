#include <iostream>
#include <cstring>
#include <SDL2/SDL.h>
#include <mandelbrot_highway.h>
#include <mandelbrot_opencl.h>
#include <CL/cl.h>
#include <common_math.h>
#include <easycl.h>
#include <thread>

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
    int num_devices = omp_get_num_devices();
    #pragma omp target
    {
        if (omp_is_initial_device())
        {
            printf("No GPU detected by OpenMP\n");
        }
        else
        {
            int nteams = omp_get_num_teams();
            int nthreads = omp_get_num_threads();
            printf("OpenMP: Running on device with %d teams in total and %d threads in each team\n", nteams, nthreads);
        }
    }

    int acc_devices = acc_get_num_devices(acc_device_nvidia);
    if (acc_devices > 0)
    {
        printf("OpenACC: Running on NVIDIA device\n");
    }
    else
    {
        printf("No GPU detected by OpenACC\n");
    }

    auto clDevices = easycl::getDevices(CL_DEVICE_TYPE_GPU);
    if (clDevices.size()>0)
    {
        printf("OpenCL - %d GPUs detected:\n", clDevices.size());

        for (const auto& d : clDevices)
        {
            std::cout << d.getName() << "\n";
        }
    }

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
            0, window_w, window_h, 32, SDL_PIXELFORMAT_RGBA8888);

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
    //time_and_test(howMany, mandelbrot_base<float>,                 "Base algo single core ", buffer, window_w, window_h);
    time_and_test(howMany, mandelbrot_omp<float>, "Base algo on       " + std::to_string(ncpus) + " threads", buffer, window_w, window_h);
    if (num_devices > 0)
    {
        time_and_test(howMany, mandelbrot_omp_gpu<float>, "Base algo OMP/GPU  ", buffer, window_w, window_h);
    }

    if (acc_devices > 0)
    {
        time_and_test(howMany, mandelbrot_acc_gpu<float>, "Base algo ACC/GPU  ", buffer, window_w, window_h);
    }

    for (auto d : clDevices)
    {
        d.loadProgram("../src/mandelbrot_opencl_program.cl", "calc_pixel");
        auto mandelbrotToCall = [&](void *points, int w, int h){
            mandelbrot_opencl(d, points, w, h);
        };
        time_and_test(howMany, mandelbrotToCall, d.getName(), buffer, window_w, window_h);
    }

    time_and_test(howMany, mandelbrot_highway::mandelbrot, "Auto dispatch on   " + std::to_string(ncpus) + " threads", buffer, window_w, window_h);
    time_and_test(howMany, mandelbrot_highway::mandelbrot_sse2, "SSE2 dispatch on   " + std::to_string(ncpus) + " threads", buffer, window_w, window_h);
    time_and_test(howMany, mandelbrot_highway::mandelbrot_ssse3, "SSSE3 dispatch on  " + std::to_string(ncpus) + " threads", buffer, window_w, window_h);
    time_and_test(howMany, mandelbrot_highway::mandelbrot_sse4, "SSE4 dispatch on   " + std::to_string(ncpus) + " threads", buffer, window_w, window_h);
    time_and_test(howMany, mandelbrot_highway::mandelbrot_avx2, "AVX2 dispatch on   " + std::to_string(ncpus) + " threads", buffer, window_w, window_h);
    // time_and_test(howMany, mandelbrot_highway::mandelbrot_avx512, "AVX512 dispatch on " + std::to_string(ncpus) + " threads", buffer, window_w, window_h);

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
