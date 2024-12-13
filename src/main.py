import PIL.Image, time
import mandelbrot_wrapper as w
from ctypes import * 


x = 4096
y = 4096

# create a buffer
imagebuffer1 = w.makebuffer(x, y)
mandel_cpu = w.mandel(use_gpu = False)
mandel_gpu = w.mandel(use_gpu = True)

print("benchmarking CPU")

start = time.time_ns()
for i in range(5):
    mandel_cpu.run(imagebuffer1, x, y)
elapsed = (time.time_ns() - start) / 5000000
print(f"average over 5 runs {elapsed}ms")

PIL.Image.frombuffer(size=(x,y), data=imagebuffer1, mode="RGBA", decoder_name="raw").show()

imagebuffer2 = w.makebuffer(x, y)

print("benchmarking GPU")
start = time.time_ns()
for i in range(5):
    mandel_gpu.run(imagebuffer2, x, y)
elapsed = (time.time_ns() - start) / 5000000
print(f"average over 5 runs {elapsed}ms")

PIL.Image.frombuffer(size=(x,y), data=imagebuffer2, mode="RGBX").show()

