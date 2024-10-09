#include <iostream>
#include <cstring>
#include <SDL2/SDL.h>
#include <mandelbrot_highway.h>
#include <common_math.h>
#include <thread>

constexpr size_t window_w = 1024;
constexpr size_t window_h = 1024;

using namespace mandelbrot_highway;

void time_and_test(size_t iterations, MandelbrotFunc f, const std::string& name, void * buffer, int w, int h)
{
    using ns = std::chrono::nanoseconds;
    ns bestTime{std::chrono::nanoseconds::max()};
    ns worst{std::chrono::nanoseconds::min()};

    for (int i=0; i<iterations; ++i)
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

    std::cout << "Testing "<< name << " over " << iterations << " iterations: ";
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(bestTime) << " - " << std::chrono::duration_cast<std::chrono::milliseconds>(worst) << "\n";
}

int main(int argc, char** argv)
{
    bool render = true;
    if (argc == 2 && strcmp(argv[1], "-norender") == 0) render = false;
    void * buffer;
    SDL_PixelFormat *format = nullptr;
    SDL_Surface *canvas = nullptr;
    SDL_Surface *s = nullptr;
    SDL_Window *window = nullptr;
    
    if (render)
    {
        // Init video
        SDL_Init(SDL_INIT_VIDEO);

        window = SDL_CreateWindow("SDL2Test",SDL_WINDOWPOS_UNDEFINED,SDL_WINDOWPOS_UNDEFINED,window_w,window_h,0);
        if (!window) return 0;

        // Calculate shit
        s = SDL_GetWindowSurface(window);
        canvas = SDL_CreateRGBSurfaceWithFormat(
            0, window_w, window_h, 32, SDL_PIXELFORMAT_RGBA8888
        );

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
    // TODO: Make something appear on the screen while calculating
    // std::thread t;
    // if (render)
    // {
    //     // Start background thread to update the graphics
    //     t = std::thread([&]{
    //         while (!quit)
    //         {
    //             SDL_Delay(500);

    //             if (SDL_MUSTLOCK(canvas))
    //             {
    //                 SDL_LockSurface(canvas);
    //             }

    //             SDL_BlitSurface(canvas, 0, s, 0);
    //             SDL_UpdateWindowSurface(window);

    //             if (SDL_MUSTLOCK(canvas))
    //             {
    //                 SDL_UnlockSurface(canvas);
    //             }

    //         }
    //     });
    // }

    const int howMany = 5;
    time_and_test(howMany, mandelbrot_base<float>,          "Base algo single core ", buffer, window_w, window_h);
    time_and_test(howMany, mandelbrot_omp<float>,           "Base algo on 4 core   ", buffer, window_w, window_h);
    time_and_test(howMany, mandelbrot_highway::mandelbrot,  "AVX2 algo on 4 core   ", buffer, window_w, window_h);
    //t.join();

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
}

