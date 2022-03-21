"""
Helper functions.

"""
import os
import numpy as np
import tensorrt as trt


def get_engine(engine_file_path=None):
     if os.path.exists(engine_file_path):
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(trt.Logger(trt.Logger.WARNING)) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
            return runtime.deserialize_cuda_engine(f.read())


# Loads a test case into the provided pagelocked_buffer.
def load_normalized_test_case(fp, pagelocked_buffer):
    # Flatten into a 1D array, and copy to pagelocked memory.
    input_batch = np.load(fp)
    np.copyto(pagelocked_buffer, input_batch.ravel())


def load_normalized_test_case_v2(fp, pagelocked_buffer):
    # Flatten into a 1D array, and copy to pagelocked memory.
    input_batch = np.load(fp)
    pagelocked_buffer[:] = input_batch.ravel()


def visualize(input_batch):
    # set up the figure
    fig = plt.figure(figsize=(15, 7))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    # plot the images: each image is 28x28 pixels
    for i in range(input_batch.shape[0]):
        ax = fig.add_subplot(5, 10, i + 1, xticks=[], yticks=[])
        ax.imshow(input_batch[i,:].reshape((28,28)).astype(np.float32),cmap=plt.cm.gray_r, interpolation='nearest')
