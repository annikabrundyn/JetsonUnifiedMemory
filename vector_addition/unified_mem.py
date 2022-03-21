""" Vector Addition Example:

Using CUDA Unified Memory.

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
    a = cuda.managed_empty(shape=1 << 24, dtype=np.float32, mem_flags=cuda.mem_attach_flags.GLOBAL)
    b = cuda.managed_empty(shape=1 << 24, dtype=np.float32, mem_flags=cuda.mem_attach_flags.GLOBAL)
    a[:] = np.random.randn(1 << 24).astype('float32')
    b[:] = np.random.randn(1 << 24).astype('float32')
    c = cuda.managed_zeros(shape=1 << 24, dtype=np.float32, mem_flags=cuda.mem_attach_flags.GLOBAL)

    # Create the kernel
    mod = SourceModule(kernel)
    func = mod.get_function('mykern')

    # GPU performance Loop
    start_gpu_calc = time.time()
    for _ in range(iters):
        func(c, a, b, np.int32(1 << 24),
             grid=((1 << 24)//256,1),
             block=(256,1,1))

        # have to synchronize (prev. implicit in memcopy call)
        pycuda.autoinit.context.synchronize()
    total_gpu_time = time.time() - start_gpu_calc

    print('--- Results for UM implementation ---')
    print('GPU implementation (total) took {:.5f}s'.format(total_gpu_time))
    print('Gpu implementation (repeated) took on average {:.5f}s'.format(total_gpu_time/iters))
    print(' ----------- ')
