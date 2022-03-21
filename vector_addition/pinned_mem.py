""" Vector Addition Example:

Using pinned/page-locked memory. Note this is specifically written for Jetson.

"""
import numpy as np
import time
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule


kernel = """
extern "C"
__global__ void mykern(float *dst, const float *a, const float *b, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        dst[i] = a[i] + b[i];
}
"""


if __name__ == '__main__':

    iters = 100

    # Setup problem
    a = cuda.pagelocked_empty(shape=1 << 24, dtype=np.float32)
    b = cuda.pagelocked_empty(shape=1 << 24, dtype=np.float32)
    a[:] = np.random.randn(1 << 24).astype('float32')
    b[:] = np.random.randn(1 << 24).astype('float32')
    c = cuda.pagelocked_zeros(shape=1 << 24, dtype=np.float32)

    # Manually get pointers usable in device code
    a_d = np.intp(a.base.get_device_pointer())
    b_d = np.intp(b.base.get_device_pointer())
    c_d = np.intp(c.base.get_device_pointer())
    
    # Create the kernel and call it
    mod = SourceModule(kernel)
    func = mod.get_function('mykern')

    # GPU Performance Loop
    start_gpu_calc = time.time()
    for _ in range(iters):
        func(c_d, a_d, b_d, np.int32(1 << 24),
             grid=((1 << 24)//256,1),
             block=(256,1,1))
        pycuda.autoinit.context.synchronize()
    total_gpu_time = time.time() - start_gpu_calc

    print('--- Results for Pinned implementation ---')
    print('GPU implementation (total) took {:.5f}s'.format(total_gpu_time))
    print('Gpu implementation (repeated) took on average {:.5f}s'.format(total_gpu_time/iters))
    print(' ----------- ')
