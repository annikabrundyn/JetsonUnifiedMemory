# Example: Inference with TensorRT

In this example we perform inference with a TensorRT optimized neural network using the TensorRT Python API.

### Note: In order to run this example you will first need to obtain a TensorRT engine.

The notebook in the `extra` folder walks through training a simple neural network in Tensorflow, exporting it to ONNX format and then converting the saved model to a TensorRT engine file. This final step (converting an ONNX file to a TensorRT engine) needs to be run on the target hardware (i.e. in this case a Jetson module). In the notebook you'll see that I trained a small convolutional network to predict on images from the Fashion MNIST dataset. For convenience, I've also saved a single batch of Fashion MNIST images as a Numpy array (`np_batch.npy`). 

We compare three different implementations:

(1) Original version written for a discrete GPU setup from the TensorRT Github repo. Reference code: https://github.com/NVIDIA/TensorRT/tree/main/samples/python/end_to_end_tensorflow_mnist

(2) Adapt to use CUDA Unified Memory - applicable to both Jetson and discrete GPU setup.

(3) Adapt to use CUDA Pinned Memory - specifically implemented for Jetson. 
