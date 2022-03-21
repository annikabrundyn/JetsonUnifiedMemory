# TensorFlow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
print("Tensorflow version: ", tf.version.VERSION)

# remember the tf version for further use
TF_VER = tf.version.VERSION.split(".")[0]

# Other supporting libraries
import os
import datetime
import matplotlib.pyplot as plt
import numpy as np
import time
import math
from tensorflow.python.compiler.tensorrt import trt_convert as trt 
import tensorrt


def visualize_batch(rnd_idx, batch_size, predictions = False):
    
    subplot_rows = math.ceil(batch_size / 10)
    subplot_cols = 10
    
    # set up the figure
    fig = plt.figure(figsize=(15, subplot_rows * 1.5))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    # plot the images: each image is 28x28 pixels
    j = 0
    for i in range(rnd_idx, rnd_idx + batch_size):
        ax = fig.add_subplot(subplot_rows, subplot_cols, j + 1, xticks=[], yticks=[])
        ax.imshow(test_images[i,:].reshape((28,28)),cmap=plt.cm.gray_r, interpolation='nearest')
        if predictions is False:
            ax.text(0, 7, class_names[np.nonzero(test_labels[i])[0][0]], color='green')
        else:
            if predictions[j] == class_names[np.nonzero(test_labels[i])[0][0]]:
                ax.text(0, 7, predictions[j], color='blue')
            else:
                ax.text(0, 7, predictions[j], color='red')
        j += 1


def run_unoptimized(model, loop_count=200, warmup_runs=50):

    for i in range(warmup_runs):
        prediction = model.predict(x=item_image2)

    start_time = time.time()
    for i in range(loop_count):
        prediction = model.predict(x=item_image2)

    print("Unoptimized inference time: %s ms in average" %((time.time() - start_time) * 1000 / loop_count))

    predicted_labels = [class_names[np.argmax(prediction, axis=-1)[i]] for i in range(len(prediction))]

    visualize_batch(image_index, BATCH_SIZE, predicted_labels)


def run_trt_test(loaded_model, loop_count = 200, warmup_inter = 50):

    for i in range(warmup_inter):
        prediction = loaded_model(tf.constant(random_batch))
    
    start_time = time.time()
    for i in range(loop_count):
        prediction = loaded_model(tf.constant(random_batch))

    print("TF-TRT inferences with %s ms in average" %((time.time() - start_time) * 1000 / loop_count))
    
    predicted_labels = [class_names[np.argmax(prediction, axis=-1)[i]] for i in range(len(prediction))]

    visualize_batch(idx, BATCH_SIZE, predicted_labels)


def load_data():
    fashion_mnist = keras.datasets.fashion_mnist

    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    # Add a new axis
    train_images = train_images[:, :, :, np.newaxis]
    test_images = test_images[:, :, :, np.newaxis]

    # Convert class vectors to binary class matrices.
    train_labels = to_categorical(train_labels, num_classes)
    test_labels = to_categorical(test_labels, num_classes)

    # Data normalization
    train_images = train_images.astype('float32')
    test_images = test_images.astype('float32')
    train_images /= 255.
    test_images /= 255.
    
    #Note: TensorRT only supports 'channels first' input data type
    train_images = np.rollaxis(train_images, 3, 1) 
    test_images = np.rollaxis(test_images, 3, 1) 
    
    return train_images, train_labels, test_images, test_labels


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

num_classes = len(class_names)

# Load data
train_images, train_labels, test_images, test_labels = load_data()


MODEL_DIR = "./models" 
TMP_NUMBER_PATH = os.path.join(MODEL_DIR, "rnd_idx.txt")
OUT_MODEL_PATH = os.path.join(MODEL_DIR, 'saved_model')
FROZEN_GRAPH_PATH_TF1 = os.path.join(MODEL_DIR, "frozen_graph.pb")
FROZEN_GRAPH_PATH_TF2 = os.path.join(MODEL_DIR, "frozen_graph_tf2.pb")
TRT_OUTPUT_PATH = os.path.join(MODEL_DIR, "optimized_model")
TRT_OUTPUT_PATH_TFTRT_FP16 = os.path.join(MODEL_DIR, "TFTRT_FP16")
TRT_OUTPUT_PATH_TFTRT_INT8 = os.path.join(MODEL_DIR, "TFTRT_INT8")
TRT_ENGINE_PATH = "models/out_model.engine"
TRT_LOGGER = tensorrt.Logger(tensorrt.Logger.VERBOSE)
ONNX_TRT_ENGINE_PATH = MODEL_DIR + "/trt_onnx_out.plan"
OUT_ONNX_MODEL = MODEL_DIR + "/model.onnx"


# Load from file the saved random index.
idx = 0
with open(TMP_NUMBER_PATH, 'r') as f:
    idx = int(f.readline())
    BATCH_SIZE = int(f.readline())
    TF_VER_used_for_train = int(f.readline())
    print(idx)
    print(BATCH_SIZE)
    print(TF_VER_used_for_train)

random_batch = test_images[idx:(idx + BATCH_SIZE)]
random_batch_labels = test_labels[idx:(idx + BATCH_SIZE)]
