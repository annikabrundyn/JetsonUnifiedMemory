# Using Unified Memory on Jetson

**Note: this repository accompanies a talk I presented at GTC Spring 2022: Demystifying Unified Memory on Jetson [SE2600]**

On Jetson, because device memory, host memory, and unified memory are allocated on the same physical SoC DRAM, duplicate memory allocations and data transfers can be avoided. CUDA applications can use various kinds of memory buffers, such as device memory, pageable host memory, pinned memory, and unified memory. Even though these memory buffer types are allocated on the same physical device, each has different accessing and caching behaviors. This repository contains two examples of how to adapt your PyCUDA code written for a discrete GPU setup for running on Jetson using either unified memory or pinned memory:

(1) Simple vector addition

(2) Inference with a TensorRT optimized neural network.




### Setup:

Docker container used: nvcr.io/nvidia/l4t-ml:r32.6.1-py3

Available at: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-ml

Tested on a Jetson Xavier NX Developer Kit.
