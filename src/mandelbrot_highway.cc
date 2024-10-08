
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "../src/mandelbrot_highway.cc" // this file
#include <mandelbrot_highway.h>
#include <hwy/foreach_target.h>  // must come before highway.h
#include <hwy/highway.h>
#include <SDL2/SDL.h>
#include <common_math.h>
#include <type_traits>

HWY_BEFORE_NAMESPACE();
namespace mandelbrot_highway {
namespace HWY_NAMESPACE {  // required: unique per target

    using T = float;

    template <class Vec>
    decltype(auto) vec_map_to(Vec value, Vec sourceMin, Vec sourceMax, Vec destMin, Vec destMax)
    {
        auto s = Sub(sourceMax, sourceMin);
        auto d = Sub(destMax, destMin);
        auto sourceRatio = Div(Sub(value, sourceMin), s);
        return MulAdd(sourceRatio, d, destMin);
    }

    void mandelbrot_vec_omp(void* HWY_RESTRICT points,
                    int w, int h, const SDL_PixelFormat* format) 
    {
        using namespace hwy::HWY_NAMESPACE;
        uint32_t * p = (uint32_t *)points; 
        const ScalableTag<T> d;
        using V = decltype(Zero(d));

         // Assuming the image width is an integer multiple of the num of lanes in the SIMD unit 
        SDL_assert(w % Lanes(d) == 0);
        
        V oneTwoThreeFour = Zero(d);
        for (size_t lane=0; lane<Lanes(d); ++lane)
        {
            oneTwoThreeFour = InsertLane<T>(oneTwoThreeFour, lane, T{1} * lane);
        }

        #pragma omp parallel for default(private) shared(h,w,oneTwoThreeFour,p,format)
        for (int j=0; j<h; ++j)
        {
            for (int i=0; i<w; i += Lanes(d))
            {
                V x0 = vec_map_to<V>(Set(d,i) + oneTwoThreeFour, Zero(d), Set(d,w), Set(d, -2.0), Set(d, 0.47));
                V y0 = vec_map_to<V>(Set(d,j), Zero(d), Set(d,h), Set(d, -1.12), Set(d, 1.12));
                V x = Zero(d);
                V y = Zero(d);
                V iteration = Zero(d);
                const V max_iteration = Set(d, 1000);

                V ySquared = Mul(y ,y);

                const V one = Set(d, 1.0);
                const V two = Set(d, 2.0);
                const V four = Set(d, 4.0);
                const V onethousands = Set(d, 1000.0);

                auto LessThan4 = x*x + y*y <= four;
                bool keepChurning = CountTrue(d, LessThan4) >= 1;

                int scalariteration = 0;
                while (keepChurning && scalariteration < 1000)
                {
                    V xTemp = x*x - y*y + x0;
                    y = two * x * y + y0;
                    x = xTemp;
                    
                    LessThan4 = x*x + y*y <= four;
                    iteration = MaskedAddOr(iteration, LessThan4, iteration, one);
                    keepChurning = CountTrue(d, LessThan4) >= 1;
                    ++scalariteration;
                }

                V color = vec_map_to<V>(iteration, Zero(d), Set(d, 15.0), Zero(d), Set(d, 255));

                // Map a vector of colors (slow, non vectorized)
                // std::array<T, Lanes(d)> vColor;
                // Store(color, d, vColor.data());

                for (size_t lane=0; lane<Lanes(d); ++lane)
                {
                    Uint8 value = (Uint8)color.raw[lane];
                    p[j*w + i + lane] = SDL_MapRGBA(format, value, value, value, 255);
                }
            }
        }
    }


} // HWY_NAMESPACE
} // mandelbrot_highway
HWY_AFTER_NAMESPACE();



// The table of pointers to the various implementations in HWY_NAMESPACE must
// be compiled only once (foreach_target #includes this file multiple times).
// HWY_ONCE is true for only one of these 'compilation passes'.
#if HWY_ONCE



namespace mandelbrot_highway {
    // This macro declares a static array used for dynamic dispatch

    //HWY_EXPORT(mandelbrot_vec);    
    HWY_EXPORT(mandelbrot_vec_omp);

    void mandelbrot(void* HWY_RESTRICT points, int w, int h, const SDL_PixelFormat* format) 
    {
        return HWY_DYNAMIC_DISPATCH(mandelbrot_vec_omp)(points,w,h,format);
    }

}
#endif