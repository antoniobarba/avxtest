#include <iostream>
#include <cstring>
#include <SDL2/SDL.h>
#include <mandelbrot_highway.h>
#include <common_math.h>
#include <thread>

constexpr size_t window_w = 1024;
constexpr size_t window_h = 1024;

using namespace mandelbrot_highway;

void time_and_test(int iterations, MandelbrotFunc f, const std::string& name, void * buffer, int w, int h, const SDL_PixelFormat* format)
{
    auto time_before = std::chrono::high_resolution_clock::now();
    for (int i=0; i<iterations; ++i) f(buffer, w, h, format);
    auto elapsed = std::chrono::high_resolution_clock::now() - time_before;
    std::cout << "name: "<< name << "\t" << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed / iterations) << " average over " << iterations << " iterations\n";
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

    const int howMany = 3;
    time_and_test(howMany, mandelbrot_base<float>,          "Base algo           ", buffer, window_w, window_h, format);
    time_and_test(howMany, mandelbrot_omp<float>,           "Base algo, 4 core   ", buffer, window_w, window_h, format);
    time_and_test(howMany, mandelbrot_highway::mandelbrot,  "AVX2 algo, 4 core   ", buffer, window_w, window_h, format);
    
    if (render)
    {
        if (SDL_MUSTLOCK(canvas))
        {
            SDL_UnlockSurface(canvas);
        }

        // Cleanup
        bool quit = false;
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

