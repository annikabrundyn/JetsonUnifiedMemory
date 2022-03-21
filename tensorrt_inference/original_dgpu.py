""" Inference with TensorRT: Original/dGPU Example

Original example written for dGPU set-up from TensorRT GitHub repo.

See reference code -
https://github.com/NVIDIA/TensorRT/blob/main/samples/python/common.py
https://github.com/NVIDIA/TensorRT/blob/main/samples/python/end_to_end_tensorflow_mnist/sample.py

Uses page-locked memory, asynchronous transfer + execution

"""
import time
import argparse

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

from common import *


# Just a helper class
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def allocate_buffers_dgpu(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    
    return inputs, outputs, bindings, stream


def do_inference_dgpu(context, bindings, inputs, outputs, stream):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()


def main(warmup_iters=1000, iters=500):
    # Build an engine, allocate buffers and create a stream.
    engine = get_engine("ab_trt_engine.trt")
    inputs, outputs, bindings, stream = allocate_buffers_dgpu(engine)
    
    with engine.create_execution_context() as context:
        # load saved batch from Fashion MNIST
        load_normalized_test_case("np_batch.npy", inputs[0].host)

        # warmup
        for _ in range(warmup_iters):
            do_inference_dgpu(context, bindings, inputs, outputs, stream)

        # start timing inference
        start = time.time()
        for _ in range(iters):
            do_inference_dgpu(context, bindings, inputs, outputs, stream)
        end = time.time()

        total_inf_time = end-start
        avg_inf_time = (end-start)/iters

        print("--- dGPU original results ---")
        print("total time: ", total_inf_time)
        print("avg time: ", avg_inf_time)

            
if __name__ == '__main__':
    main()
