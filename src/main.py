import PIL.Image, time
import mandelbrot_wrapper as w
from ctypes import * 


x = 4096
y = 4096

# create a buffer
imagebuffer1 = w.makebuffer(x, y)

print("benchmarking CPU")

start = time.time_ns()
for i in range(5):
    w.mandelbrot_cpu(imagebuffer1, x, y)
elapsed = (time.time_ns() - start) / 5000000
print(f"average over 5 runs {elapsed}ms")

PIL.Image.frombuffer(size=(x,y), data=imagebuffer1, mode="RGBA", decoder_name="raw").show()

imagebuffer2 = w.makebuffer(x, y)
kernel = w.mandelbrot_create_gpu_kernel()

print("benchmarking GPU")
start = time.time_ns()
for i in range(5):
    w.mandelbrot_gpu(kernel, imagebuffer2, x, y)
elapsed = (time.time_ns() - start) / 5000000
print(f"average over 5 runs {elapsed}ms")

w.mandelbrot_free_gpu_kernel(kernel)
PIL.Image.frombuffer(size=(x,y), data=imagebuffer2, mode="RGBX").show()

