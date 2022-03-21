We implement a element-wise addition of two large vectors `a` and `b` to produce a resulting vector `c`. Each example uses a different type of CUDA memory allocation:

(1) Original discrete GPU (dGPU) example - i.e. contains both a host and device memory allocation.

(2) Using CUDA Unified Memory. 

(3) Using CUDA Pinned/Page-locked Memory.
