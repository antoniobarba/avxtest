from ctypes import *

mandelbrot = None
try:
    mandelbrot = cdll.LoadLibrary("libmandelbrot.so")
except:
    print('libmandelbrot.so not found, please verify that it is installed and/or set the LD_LIBRARY_PATH env variable to where it is located.')
    print('Example: LD_LIBRARY_PATH="/some/awesome/directory" python YourProgramUsingLibMandelbrot.py')
    exit(1)

def makebuffer(w, h):
    return (c_int32 * w * h)()

class mandel:
    def __init__(self, use_gpu: bool):
        if use_gpu:
            self.kernel = mandelbrot_create_gpu_kernel()
        else:
            self.kernel = None

    def __del__(self):
        if self.kernel is not None:
            mandelbrot_free_gpu_kernel(self.kernel)

    def run(self, buffer, width: int, height: int):
        if self.kernel is not None:
            mandelbrot_gpu(self.kernel, buffer, width, height)
        else:
            mandelbrot_cpu(buffer, width, height)

# points must be allocated with at least 4 x w x h bytes 
# extern void mandelbrot_cpu(void * points, int w, int h);
__mandelbrot_cpu_proto = CFUNCTYPE(None, c_void_p, c_int, c_int)
__mandelbrot_cpu_paramflags = (1,"points"),(1,"w"),(1,"h")
mandelbrot_cpu = __mandelbrot_cpu_proto(("mandelbrot_cpu", mandelbrot), __mandelbrot_cpu_paramflags)

# points must be allocated with at least 4 x w x h bytes 
# extern void mandelbrot_gpu(void * kernel, void * points, int w, int h);
__mandelbrot_gpu_proto = CFUNCTYPE(None, c_void_p, c_void_p, c_int, c_int)
__mandelbrot_gpu_paramflags = (1,"kernel"),(1,"points"),(1,"w"),(1,"h")
mandelbrot_gpu = __mandelbrot_gpu_proto(("mandelbrot_gpu", mandelbrot), __mandelbrot_gpu_paramflags)

# extern void * mandelbrot_create_gpu_kernel();
__mandelbrot_create_gpu_kernel_proto = CFUNCTYPE(c_void_p)
mandelbrot_create_gpu_kernel = __mandelbrot_create_gpu_kernel_proto(("mandelbrot_create_gpu_kernel", mandelbrot), None)

# extern void mandelbrot_free_gpu_kernel(void *kernel);
__mandelbrot_free_gpu_kernel_proto = CFUNCTYPE(None, c_void_p)
__mandelbrot_free_gpu_kernel_paramflags = (1,"kernel"),
mandelbrot_free_gpu_kernel = __mandelbrot_free_gpu_kernel_proto(("mandelbrot_free_gpu_kernel", mandelbrot), __mandelbrot_free_gpu_kernel_paramflags)

