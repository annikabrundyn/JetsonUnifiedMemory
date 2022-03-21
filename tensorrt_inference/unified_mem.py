""" Inference with TensorRT: Unified Memory

Adapt original TensorRT Example to use CUDA Unified Memory.

"""
import time
import argparse

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

from common import *


def allocate_buffers_um(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        
        # Allocate single managed mem buffer
        mem = cuda.managed_empty(size, dtype, mem_flags=cuda.mem_attach_flags.GLOBAL)
        
        # Append the device buffer to device bindings.
        bindings.append(int(mem.base.get_device_pointer()))
        
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(mem)
        else:
            outputs.append(mem)
    
    return inputs, outputs, bindings, stream


def do_inference_um(context, bindings, stream):
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Now that we force synchronization here this is slow too...
    stream.synchronize()


def main(warmup_iters=1000, iters=500):
    # Build an engine, allocate buffers and create a stream.
    engine = get_engine("ab_trt_engine.trt")
    inputs, outputs, bindings, stream = allocate_buffers_um(engine)

    with engine.create_execution_context() as context:
        # load saved batch from Fashion MNIST
        load_normalized_test_case("np_batch.npy", inputs[0])

        # warmup
        for _ in range(warmup_iters):
            do_inference_um(context, bindings, stream)

        # start timing inference
        start = time.time()
        for _ in range(iters):
            do_inference_um(context, bindings, stream)
        end = time.time()

        total_inf_time = end - start
        avg_inf_time = (end - start) / iters

        print("--- UM/MM version results ---")
        print("total time: ", total_inf_time)
        print("avg time: ", avg_inf_time)


if __name__ == '__main__':
    main()
