""" Vector Addition Example: 

Original implementation for discrete GPU setup (i.e. not adapted for Jetson)

This is a simple example that shows how to add two very large arrays.

"""
import numpy as np
import time
import pycuda.autoinit  # noqa - Have to import otherwise won't run.
import pycuda.driver as cuda
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
    a = np.random.randn(1 << 24).astype('float32')
    b = np.random.randn(1 << 24).astype('float32')
    c2 = np.zeros(1 << 24, dtype='float32')

    # Compare to CPU-only Numpy implementation (won't repeat this in other examples)
    start_np = time.time()
    c = a + b
    np_time = time.time() - start_np
    print('Numpy implementation took {:.5f}s'.format(np_time))

    # Create the kernel
    mod = SourceModule(kernel)
    func = mod.get_function('mykern')

    # Allocate device memory - shouldn't be included in the performance loop
    start_alloc = time.time()
    a_gpu = cuda.mem_alloc(a.nbytes)
    b_gpu = cuda.mem_alloc(b.nbytes)
    c_gpu = cuda.mem_alloc(b.nbytes)
    mem_alloc_time = time.time() - start_alloc

    ### GPU Peformance Loop
    start_gpu_calc = time.time()
    for _ in range(iters):
        # Copy inputs CPU --> GPU
        cuda.memcpy_htod(a_gpu, a)
        cuda.memcpy_htod(b_gpu, b)

        # Create the kernel and call it
        func(c_gpu, a_gpu, b_gpu, np.int32(1 << 24),
             grid=((1 << 24)//256,1), block=(256,1,1))

        # Copy results GPU --> CPU (forced synchronization call)
        cuda.memcpy_dtoh(c2, c_gpu)
    total_gpu_time = time.time() - start_gpu_calc

    # Free memory allocations (alternatively import pycuda.autoinit)
    a_gpu.free()
    b_gpu.free()
    c_gpu.free()

    # Sanity check
    np.testing.assert_array_almost_equal(c, c2)

    print('--- Results for dGPU implementation ---')
    print('Memory Allocation time took {:.5f}s:'.format(mem_alloc_time))
    print('GPU implementation (total) took {:.5f}s'.format(total_gpu_time))
    print('Gpu implementation (repeated) took on average {:.5f}s'.format(total_gpu_time/iters))
    print(' ----------- ')
